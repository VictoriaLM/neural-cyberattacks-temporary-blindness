/*
 *  simulation_manager.cpp
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "simulation_manager.h"

// C includes:
#include <sys/time.h>

// C++ includes:
#include <limits>
#include <vector>

#include <fstream>
#include <iostream>
#include <string>
#include <random>
#include <unordered_set>


// Includes from libnestutil:
#include "compose.hpp"
#include "numerics.h"

// Includes from nestkernel:
#include "connection_manager_impl.h"
#include "event_delivery_manager.h"
#include "kernel_manager.h"

// Includes from sli:
#include "dictutils.h"

#include "nest.h"

#include <thread>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <mpi.h>

using json = nlohmann::json;

nest::SimulationManager::SimulationManager()
  : clock_( Time::tic( 0L ) )
  , slice_( 0L )
  , to_do_( 0L )
  , to_do_total_( 0L )
  , from_step_( 0L )
  , to_step_( 0L ) // consistent with to_do_ == 0
  , t_real_( 0L )
  , prepared_( false )
  , simulating_( false )
  , simulated_( false )
  , inconsistent_state_( false )
  , print_time_( false )
  , use_wfr_( true )
  , wfr_comm_interval_( 1.0 )
  , wfr_tol_( 0.0001 )
  , wfr_max_iterations_( 15 )
  , wfr_interpolation_order_( 3 )
  , update_time_limit_( std::numeric_limits< double >::infinity() )
  , min_update_time_( std::numeric_limits< double >::infinity() )
  , max_update_time_( -std::numeric_limits< double >::infinity() )
  , eprop_update_interval_( 1000. )
  , eprop_learning_window_( 1000. )
  , eprop_reset_neurons_on_update_( true )
  , type_attack()
  , CONN_FLO_dict()
  , CONN_JAM_dict()
  , neuron_dict()
{
}

void
nest::SimulationManager::initialize( const bool adjust_number_of_threads_or_rng_only )
{
  if ( adjust_number_of_threads_or_rng_only )
  {
    return;
  }

  Time::reset_to_defaults();
  Time::reset_resolution();

  clock_.set_to_zero();
  clock_.calibrate();

  to_do_ = 0;
  to_do_total_ = 0;
  slice_ = 0;
  from_step_ = 0;
  to_step_ = 0; // consistent with to_do_ = 0
  t_real_ = 0;

  prepared_ = false;
  simulating_ = false;
  simulated_ = false;
  inconsistent_state_ = false;
  print_time_ = false;
  use_wfr_ = true;

  wfr_comm_interval_ = 1.0;
  wfr_tol_ = 0.0001;
  wfr_max_iterations_ = 15;
  wfr_interpolation_order_ = 3;
  update_time_limit_ = std::numeric_limits< double >::infinity();
  min_update_time_ = std::numeric_limits< double >::infinity();
  max_update_time_ = -std::numeric_limits< double >::infinity();

  reset_timers_for_preparation();
  reset_timers_for_dynamics();
}

void
nest::SimulationManager::finalize( const bool )
{
}

void
nest::SimulationManager::reset_timers_for_preparation()
{
  sw_communicate_prepare_.reset();
#ifdef TIMER_DETAILED
  sw_gather_target_data_.reset();
#endif
}

void
nest::SimulationManager::reset_timers_for_dynamics()
{
  sw_simulate_.reset();
#ifdef TIMER_DETAILED
  sw_gather_spike_data_.reset();
  sw_gather_secondary_data_.reset();
  sw_update_.reset();
  sw_deliver_spike_data_.reset();
  sw_deliver_secondary_data_.reset();
#endif
}

void
nest::SimulationManager::set_status( const DictionaryDatum& d )
{
  // Create an instance of time converter here to capture the current
  // representation of time objects: TICS_PER_MS and TICS_PER_STEP
  // will be stored in time_converter.
  // This object can then be used to convert times in steps
  // (e.g. Connection::delay_) or tics to the new representation.
  // We pass this object to ConnectionManager::calibrate to update
  // all time objects in the connection system to the new representation.
  // MH 08-04-14
  TimeConverter time_converter;

  double time;
  if ( updateValue< double >( d, names::biological_time, time ) )
  {
    if ( time != 0.0 )
    {
      throw BadProperty( "The simulation time can only be set to 0.0." );
    }

    if ( clock_ > TimeZero )
    {
      // reset only if time has passed
      LOG( M_WARNING,
        "SimulationManager::set_status",
        "Simulation time reset to t=0.0. Resetting the simulation time is not "
        "fully supported in NEST at present. Some spikes may be lost, and "
        "stimulation devices may behave unexpectedly. PLEASE REVIEW YOUR "
        "SIMULATION OUTPUT CAREFULLY!" );

      clock_ = Time::step( 0 );
      from_step_ = 0;
      slice_ = 0;
      // clear all old spikes
      kernel().event_delivery_manager.configure_spike_data_buffers();
    }
  }

  updateValue< bool >( d, names::print_time, print_time_ );

  // tics_per_ms and resolution must come after local_num_thread /
  // total_num_threads because they might reset the network and the time
  // representation
  double tics_per_ms = 0.0;
  bool tics_per_ms_updated = updateValue< double >( d, names::tics_per_ms, tics_per_ms );
  double resd = 0.0;
  bool res_updated = updateValue< double >( d, names::resolution, resd );

  if ( tics_per_ms_updated or res_updated )
  {
    std::vector< std::string > errors;
    if ( kernel().node_manager.size() > 0 )
    {
      errors.push_back( "Nodes have already been created" );
    }
    if ( has_been_simulated() )
    {
      errors.push_back( "Network has been simulated" );
    }
    if ( kernel().model_manager.are_model_defaults_modified() )
    {
      errors.push_back( "Model defaults were modified" );
    }

    if ( errors.size() == 1 )
    {
      throw KernelException( errors[ 0 ] + ": time representation cannot be changed." );
    }
    if ( errors.size() > 1 )
    {
      std::string msg = "Time representation unchanged. Error conditions:";
      for ( auto& error : errors )
      {
        msg += " " + error + ".";
      }
      throw KernelException( msg );
    }

    // only allow TICS_PER_MS to be changed together with resolution
    if ( res_updated and tics_per_ms_updated )
    {
      if ( resd < 1.0 / tics_per_ms )
      {
        throw KernelException( "Resolution must be greater than or equal to one tic. Value unchanged." );
      }
      else if ( not is_integer( resd * tics_per_ms ) )
      {
        throw KernelException( "Resolution must be a multiple of the tic length. Value unchanged." );
      }
      else
      {
        const double old_res = nest::Time::get_resolution().get_ms();
        const tic_t old_tpms = nest::Time::get_resolution().get_tics_per_ms();

        nest::Time::set_resolution( tics_per_ms, resd );
        // adjust to new resolution
        clock_.calibrate();
        // adjust delays in the connection system to new resolution
        kernel().connection_manager.calibrate( time_converter );
        kernel().model_manager.calibrate( time_converter );

        std::string msg =
          String::compose( "Tics per ms and resolution changed from %1 tics and %2 ms to %3 tics and %4 ms.",
            old_tpms,
            old_res,
            tics_per_ms,
            resd );
        LOG( M_INFO, "SimulationManager::set_status", msg );

        // make sure that wfr communication interval is always greater or equal
        // to resolution if no wfr is used explicitly set wfr_comm_interval
        // to resolution because communication in every step is needed
        if ( wfr_comm_interval_ < Time::get_resolution().get_ms() or not use_wfr_ )
        {
          wfr_comm_interval_ = Time::get_resolution().get_ms();
        }
      }
    }
    else if ( res_updated ) // only resolution changed
    {
      if ( resd < Time::get_ms_per_tic() )
      {
        throw KernelException( "Resolution must be greater than or equal to one tic. Value unchanged." );
      }
      else if ( not is_integer( resd / Time::get_ms_per_tic() ) )
      {
        throw KernelException( "Resolution must be a multiple of the tic length. Value unchanged." );
      }
      else
      {
        const double old_res = nest::Time::get_resolution().get_ms();

        Time::set_resolution( resd );
        clock_.calibrate(); // adjust to new resolution
        // adjust delays in the connection system to new resolution
        kernel().connection_manager.calibrate( time_converter );
        kernel().model_manager.calibrate( time_converter );

        std::string msg = String::compose( "Temporal resolution changed from %1 to %2 ms.", old_res, resd );
        LOG( M_INFO, "SimulationManager::set_status", msg );

        // make sure that wfr communication interval is always greater or equal
        // to resolution if no wfr is used explicitly set wfr_comm_interval
        // to resolution because communication in every step is needed
        if ( wfr_comm_interval_ < Time::get_resolution().get_ms() or not use_wfr_ )
        {
          wfr_comm_interval_ = Time::get_resolution().get_ms();
        }
      }
    }
    else
    {
      throw KernelException( "Change of tics_per_ms requires simultaneous specification of resolution." );
    }
  }

  // The decision whether the waveform relaxation is used
  // must be set before nodes are created.
  // Important: wfr_comm_interval_ may change depending on use_wfr_
  bool wfr;
  if ( updateValue< bool >( d, names::use_wfr, wfr ) )
  {
    if ( kernel().node_manager.size() > 0 )
    {
      LOG( M_ERROR,
        "SimulationManager::set_status",
        "Cannot enable/disable usage of waveform relaxation after nodes have "
        "been created. Please call ResetKernel first." );
      throw KernelException();
    }
    else
    {
      use_wfr_ = wfr;
      // if no wfr is used explicitly set wfr_comm_interval to resolution
      // because communication in every step is needed
      if ( not use_wfr_ )
      {
        wfr_comm_interval_ = Time::get_resolution().get_ms();
      }
    }
  }

  // wfr_comm_interval_ can only be changed if use_wfr_ is true and before
  // connections are created. If use_wfr_ is false wfr_comm_interval_ is set to
  // the resolution whenever the resolution changes.
  double wfr_interval;
  if ( updateValue< double >( d, names::wfr_comm_interval, wfr_interval ) )
  {
    if ( not use_wfr_ )
    {
      LOG( M_ERROR,
        "SimulationManager::set_status",
        "Cannot set waveform communication interval when usage of waveform "
        "relaxation is disabled. Set use_wfr to true first." );
      throw KernelException();
    }
    else if ( kernel().connection_manager.get_num_connections() != 0 )
    {
      LOG( M_ERROR,
        "SimulationManager::set_status",
        "Cannot change waveform communication interval after connections have "
        "been created. Please call ResetKernel first." );
      throw KernelException();
    }
    else if ( wfr_interval < Time::get_resolution().get_ms() )
    {
      LOG( M_ERROR,
        "SimulationManager::set_status",
        "Communication interval of the waveform relaxation must be greater or "
        "equal to the resolution of the simulation." );
      throw KernelException();
    }
    else
    {
      LOG( M_INFO, "SimulationManager::set_status", "Waveform communication interval changed successfully. " );
      wfr_comm_interval_ = wfr_interval;
    }
  }

  // set the convergence tolerance for the waveform relaxation method
  double tol;
  if ( updateValue< double >( d, names::wfr_tol, tol ) )
  {
    if ( tol < 0.0 )
    {
      LOG( M_ERROR, "SimulationManager::set_status", "Tolerance must be zero or positive" );
      throw KernelException();
    }
    else
    {
      wfr_tol_ = tol;
    }
  }

  // set the maximal number of iterations for the waveform relaxation method
  long max_iter;
  if ( updateValue< long >( d, names::wfr_max_iterations, max_iter ) )
  {
    if ( max_iter <= 0 )
    {
      LOG( M_ERROR,
        "SimulationManager::set_status",
        "Maximal number of iterations  for the waveform relaxation must be "
        "positive. To disable waveform relaxation set use_wfr instead." );
      throw KernelException();
    }
    else
    {
      wfr_max_iterations_ = max_iter;
    }
  }

  // set the interpolation order for the waveform relaxation method
  long interp_order;
  if ( updateValue< long >( d, names::wfr_interpolation_order, interp_order ) )
  {
    if ( interp_order < 0 or interp_order == 2 or interp_order > 3 )
    {
      LOG( M_ERROR, "SimulationManager::set_status", "Interpolation order must be 0, 1, or 3." );
      throw KernelException();
    }
    else
    {
      wfr_interpolation_order_ = interp_order;
    }
  }

  // update time limit
  double t_new = 0.0;
  if ( updateValue< double >( d, names::update_time_limit, t_new ) )
  {
    if ( t_new <= 0 )
    {
      LOG( M_ERROR, "SimulationManager::set_status", "update_time_limit > 0 required." );
      throw KernelException();
    }

    update_time_limit_ = t_new;
  }

  // eprop update interval
  double eprop_update_interval_new = 0.0;
  if ( updateValue< double >( d, names::eprop_update_interval, eprop_update_interval_new ) )
  {
    if ( eprop_update_interval_new <= 0 )
    {
      LOG( M_ERROR, "SimulationManager::set_status", "eprop_update_interval > 0 required." );
      throw KernelException();
    }

    eprop_update_interval_ = eprop_update_interval_new;
  }

  // eprop learning window
  double eprop_learning_window_new = 0.0;
  if ( updateValue< double >( d, names::eprop_learning_window, eprop_learning_window_new ) )
  {
    if ( eprop_learning_window_new <= 0 )
    {
      LOG( M_ERROR, "SimulationManager::set_status", "eprop_learning_window > 0 required." );
      throw KernelException();
    }
    if ( eprop_learning_window_new > eprop_update_interval_ )
    {
      LOG( M_ERROR, "SimulationManager::set_status", "eprop_learning_window <= eprop_update_interval required." );
      throw KernelException();
    }

    eprop_learning_window_ = eprop_learning_window_new;
  }

  updateValue< bool >( d, names::eprop_reset_neurons_on_update, eprop_reset_neurons_on_update_ );
}

void
nest::SimulationManager::get_status( DictionaryDatum& d )
{
  def< double >( d, names::ms_per_tic, Time::get_ms_per_tic() );
  def< double >( d, names::tics_per_ms, Time::get_tics_per_ms() );
  def< long >( d, names::tics_per_step, Time::get_tics_per_step() );
  def< double >( d, names::resolution, Time::get_resolution().get_ms() );

  def< double >( d, names::T_min, Time::min().get_ms() );
  def< double >( d, names::T_max, Time::max().get_ms() );

  def< double >( d, names::biological_time, get_time().get_ms() );
  def< long >( d, names::to_do, to_do_ );
  def< bool >( d, names::print_time, print_time_ );

  def< bool >( d, names::prepared, prepared_ );

  def< bool >( d, names::use_wfr, use_wfr_ );
  def< double >( d, names::wfr_comm_interval, wfr_comm_interval_ );
  def< double >( d, names::wfr_tol, wfr_tol_ );
  def< long >( d, names::wfr_max_iterations, wfr_max_iterations_ );
  def< long >( d, names::wfr_interpolation_order, wfr_interpolation_order_ );

  def< double >( d, names::update_time_limit, update_time_limit_ );
  def< double >( d, names::min_update_time, min_update_time_ );
  def< double >( d, names::max_update_time, max_update_time_ );

  def< double >( d, names::time_simulate, sw_simulate_.elapsed() );
  def< double >( d, names::time_communicate_prepare, sw_communicate_prepare_.elapsed() );
#ifdef TIMER_DETAILED
  def< double >( d, names::time_gather_spike_data, sw_gather_spike_data_.elapsed() );
  def< double >( d, names::time_gather_secondary_data, sw_gather_secondary_data_.elapsed() );
  def< double >( d, names::time_update, sw_update_.elapsed() );
  def< double >( d, names::time_gather_target_data, sw_gather_target_data_.elapsed() );
  def< double >( d, names::time_deliver_spike_data, sw_deliver_spike_data_.elapsed() );
  def< double >( d, names::time_deliver_secondary_data, sw_deliver_secondary_data_.elapsed() );
#endif
  def< double >( d, names::eprop_update_interval, eprop_update_interval_ );
  def< double >( d, names::eprop_learning_window, eprop_learning_window_ );
  def< bool >( d, names::eprop_reset_neurons_on_update, eprop_reset_neurons_on_update_ );
}

void
nest::SimulationManager::prepare()
{
  assert( kernel().is_initialized() );

  if ( prepared_ )
  {
    std::string msg = "Prepare called twice.";
    LOG( M_ERROR, "SimulationManager::prepare", msg );
    throw KernelException();
  }

  if ( inconsistent_state_ )
  {
    throw KernelException(
      "Kernel is in inconsistent state after an "
      "earlier error. Please run ResetKernel first." );
  }

  // reset profiling timers
  reset_timers_for_dynamics();
  kernel().event_delivery_manager.reset_timers_for_dynamics();

  t_real_ = 0;
  t_slice_begin_ = timeval(); // set to timeval{0, 0} as unset flag
  t_slice_end_ = timeval();   // set to timeval{0, 0} as unset flag

  // find shortest and longest delay across all MPI processes
  // this call sets the member variables
  kernel().connection_manager.update_delay_extrema_();
  kernel().event_delivery_manager.init_moduli();

  // if at the beginning of a simulation, set up spike buffers
  if ( not simulated_ )
  {
    kernel().event_delivery_manager.configure_spike_data_buffers();
  }

  kernel().node_manager.ensure_valid_thread_local_ids();
  kernel().node_manager.prepare_nodes();

  // we have to do enter_runtime after prepare_nodes, since we use
  // calibrate to map the ports of MUSIC devices, which has to be done
  // before enter_runtime
  if ( not simulated_ ) // only enter the runtime mode once
  {
    double tick = Time::get_resolution().get_ms() * kernel().connection_manager.get_min_delay();
    kernel().music_manager.enter_runtime( tick );
  }
  prepared_ = true;

  // check whether waveform relaxation is used on any MPI process;
  // needs to be called before update_connection_intrastructure_since
  // it resizes coefficient arrays for secondary events
  kernel().node_manager.check_wfr_use();

  if ( kernel().node_manager.have_nodes_changed() or kernel().connection_manager.connections_have_changed() )
  {
#pragma omp parallel
    {
      const size_t tid = kernel().vp_manager.get_thread_id();
      update_connection_infrastructure( tid );
    } // of omp parallel
  }
}

void
nest::SimulationManager::assert_valid_simtime( Time const& t )
{
  if ( t == Time::ms( 0.0 ) )
  {
    return;
  }

  if ( t < Time::step( 1 ) )
  {
    LOG( M_ERROR,
      "SimulationManager::run",
      String::compose( "Simulation time must be >= %1 ms (one time step).", Time::get_resolution().get_ms() ) );
    throw KernelException();
  }

  if ( t.is_finite() )
  {
    Time time1 = clock_ + t;
    if ( not time1.is_finite() )
    {
      std::string msg = String::compose(
        "A clock overflow will occur after %1 of %2 ms. Please reset network "
        "clock first!",
        ( Time::max() - clock_ ).get_ms(),
        t.get_ms() );
      LOG( M_ERROR, "SimulationManager::run", msg );
      throw KernelException();
    }
  }
  else
  {
    std::string msg = String::compose(
      "The requested simulation time exceeds the largest time NEST can handle "
      "(T_max = %1 ms). Please use a shorter time!",
      Time::max().get_ms() );
    LOG( M_ERROR, "SimulationManager::run", msg );
    throw KernelException();
  }
}

void
nest::SimulationManager::run( Time const& t )
{
  assert_valid_simtime( t );

  kernel().random_manager.check_rng_synchrony();

  if ( not prepared_ )
  {
    std::string msg = "Run called without calling Prepare.";
    LOG( M_ERROR, "SimulationManager::run", msg );
    throw KernelException();
  }

  to_do_ += t.get_steps();
  to_do_total_ = to_do_;

  if ( to_do_ == 0 )
  {
    return;
  }

  kernel().io_manager.pre_run_hook();

  // Reset local spike counters within event_delivery_manager
  kernel().event_delivery_manager.reset_counters();

  sw_simulate_.start();

  // from_step_ is not touched here.  If we are at the beginning
  // of a simulation, it has been reset properly elsewhere.  If
  // a simulation was ended and is now continued, from_step_ will
  // have the proper value.  to_step_ is set as in advance_time().
  to_step_ = std::min( from_step_ + to_do_, kernel().connection_manager.get_min_delay() );


  // Warn about possible inconsistencies, see #504.
  // This test cannot come any earlier, because we first need to compute
  // min_delay_
  // above.
  if ( t.get_steps() % kernel().connection_manager.get_min_delay() != 0 )
  {
    LOG( M_WARNING,
      "SimulationManager::run",
      "The requested simulation time is not an integer multiple of the minimal "
      "delay in the network. This may result in inconsistent results under the "
      "following conditions: (i) A network contains more than one source of "
      "randomness, e.g., two different poisson_generators, and (ii) Simulate "
      "is called repeatedly with simulation times that are not multiples of "
      "the minimal delay." );
  }
  /*
    *First of all, it is necessary to load the dictionaries.
  */
  load_dictionary_();

  call_update_();

  kernel().io_manager.post_run_hook();
  kernel().random_manager.check_rng_synchrony();

  /*
    *Once the simulation ends, the dictionaries are cleared to prepare for the next execution.
  */
  type_attack.clear();
  CONN_FLO_dict.clear();
  CONN_JAM_dict.clear();
  neuron_dict.clear();

  sw_simulate_.stop();
}

void
nest::SimulationManager::cleanup()
{
  if ( not prepared_ )
  {
    std::string msg = "Cleanup called without calling Prepare.";
    LOG( M_ERROR, "SimulationManager::cleanup", msg );
    throw KernelException();
  }

  if ( not simulated_ )
  {
    prepared_ = false;
    return;
  }

  kernel().node_manager.finalize_nodes();
  prepared_ = false;
}


/***********************************************************************/
/*
  * This function loads the dictionaries containing the attack configurations and 
  * the characteristics of neurons.
*/
void
nest::SimulationManager::load_dictionary_()
{
  std::ifstream attack_file, neuron_file, CONN_FLO_file,CONN_JAM_file;
  /*
    * The parameters for the attack dictionaries are extracted from the txt files, 
    * which indicate the attack parameters. Moreover, the characteristics of each neuron 
    * saved in the neuron_dict are from the csv file called "dictionary_new".
  */
  attack_file.open("/home/type_attack.txt");
  CONN_FLO_file.open("/home/CONN_FLO_attributes.txt");
  CONN_JAM_file.open("/home/CONN_JAM_attributes.txt");
  neuron_file.open("/home/bmtk/v1_point/dictionary_new.csv");

  if(attack_file.is_open() && CONN_FLO_file.is_open() && CONN_JAM_file.is_open() && neuron_file.is_open())
  {
    std::string text_type,text_flo,text_jam,text_pop,text_neuron, text_conn, text_comb;
    while(getline(attack_file, text_type))
    {           
      std::string key,value;
      key = text_type.substr(0,text_type.find(':'));
      value = text_type.substr(text_type.find(":") + 1);
      type_attack[key] = value;
      
    }

    while(getline(CONN_FLO_file, text_conn))
    {
      std::string key,value;

      key = text_conn.substr(0,text_conn.find(':'));
      value = text_conn.substr(text_conn.find(":") + 1);
      CONN_FLO_dict[key] = value;
    }

    while(getline(CONN_JAM_file, text_conn))
    {
      std::string key,value;

      key = text_conn.substr(0,text_conn.find(':'));
      value = text_conn.substr(text_conn.find(":") + 1);
      CONN_JAM_dict[key] = value;
    }

    getline(neuron_file, text_neuron); 
    while (getline(neuron_file, text_neuron)) {
        std::istringstream ss(text_neuron);
        std::string neuron_ids, cortical_layer, population, behaviour;

        getline(ss, neuron_ids, ';');
        getline(ss, cortical_layer, ';');
        getline(ss, population, ';');
        getline(ss, behaviour, ';');

        neuron_dict[neuron_ids] = cortical_layer + ";" + population + ";" + behaviour;
    }
  }

  attack_file.close();
  neuron_file.close();
  CONN_FLO_file.close();
  CONN_JAM_file.close();
}
/***********************************************************************/
/*
  * Function responsible for finding those neurons in the neuron dictionary that accomplish 
  * the layer and the behaviour indicated in the txt file of the attack. 
*/
std::vector<std::string>
nest::SimulationManager::find_connection_(std::string source_layer, std::string source_behaviour)
{
  /*
    * This function scans the dictionary and checks the selected parameters. When a match is found, 
    * it adds the corresponding neuron ID to a list. The output is the list of neuron IDs that satisfy 
    * the specified conditions.
  */
  std::unordered_set<std::string> unique_items;
  for(const auto& connection_pair : neuron_dict)
  {
    const std::string& connection = connection_pair.second;
    const std::string& connection_id = connection_pair.first.toString();

    size_t pos = 0, prev_pos = 0;
    std::string_view parts[4];
    int count = 0;

    while (count < 4 && (pos = connection.find(';', prev_pos)) != std::string::npos) {
        parts[count++] = std::string_view(connection).substr(prev_pos, pos - prev_pos);
        prev_pos = pos + 1;
    }
    
    if (count < 4) {
        parts[count++] = std::string_view(connection).substr(prev_pos);
    }

    std::string clean_str = std::string(parts[2]);
    clean_str.erase(std::remove(clean_str.begin(), clean_str.end(), '\r'), clean_str.end());
    clean_str.erase(std::remove(clean_str.begin(), clean_str.end(), '\n'), clean_str.end());
    if (parts[0] == source_layer && clean_str == source_behaviour)
    {
        unique_items.insert(connection_id); 
    }
  }
  return std::vector<std::string>(unique_items.begin(), unique_items.end());
}
/***********************************************************************/
/*
  * Function focused on execution the cyberattack FLO in the following attacks: FLO-FLO, JAM-FLO, and FLO-JAM.
*/
void
nest::SimulationManager::attack_based_FLO(std::string source_layer, std::string source_behaviour)
{
  /**********************STEP 1**********************/
  /*
    * As these attacks are executed using multiple cores, there was a need to implement a specific code
    * to make able multiple threading. 
  */
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  /*
    * Now, find_connection_ is used to obtain the list with the ID of the neurons that belong to the
    * specified layer and behavior, both provided as parameters.
  */
  std::vector<std::string> id_neuron_attack;
  if (rank == 0) {
    id_neuron_attack = find_connection_(source_layer, source_behaviour);  
    /*
      * Analysis of how many neurons are targeted by the attack. If `n_neurons` is 100, the cyberattack is applied 
      * to all selected neurons (100%).
    */
    std::random_device rd;
    std::mt19937 g(rd());
    int n = std::stoi(CONN_FLO_dict["n_neurons"]);
    if (n > 0 && n <= 100 && !id_neuron_attack.empty()) {
      std::shuffle(id_neuron_attack.begin(), id_neuron_attack.end(), g);
      size_t total = id_neuron_attack.size();
      size_t new_size = (total * n) / 100; 
      id_neuron_attack.resize(new_size);
    }    
  }
  /**********************STEP 2**********************/
  /*
    * It is necessary to send the list of neuron IDs (strings) from process 0 to all other MPI processes.
    * MPI cannot directly broadcast a vector<string>, so we convert it into a single char array.
  */
  /*
    * flat_data: stores all IDs together in one flat buffer of characters.
    * num_ids: store how many IDs are being sent.
  */
  std::vector<char> flat_data;
  size_t num_ids = 0;
  /*
    * Process 0 packs all strings into flat_data. After each ID, the function adds '\0' to mark the end of that string.
    * Lastly, saves how many IDs are packed.
  */
  if (rank == 0) {
    for (const auto& s : id_neuron_attack) {
      flat_data.insert(flat_data.end(), s.begin(), s.end());
      flat_data.push_back('\0');
    }
    num_ids = id_neuron_attack.size();
  }
  /*
    * At first, broadcast how many IDs will be received.
  */
  MPI_Bcast(&num_ids, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  /*
    * Then, broadcast how many characters the flat buffer contains.
  */
  size_t total_chars = flat_data.size();
  MPI_Bcast(&total_chars, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  /*
    * Processes that are not rank 0 must allocate enough space to receive the buffer.
  */
  if (rank != 0) {
    flat_data.resize(total_chars);
  }
  /*
    * Broadcast the packed char buffer itself.
  */
  MPI_Bcast(flat_data.data(), total_chars, MPI_CHAR, 0, MPI_COMM_WORLD);
  /*
    * Non-root processes rebuild the original list of strings. 
    * They read characters until '\0', which indicates the end of one ID.
  */
  if (rank != 0) {
    id_neuron_attack.clear();
    std::string tmp;
    for (char c : flat_data) {
      if (c == '\0') {
        id_neuron_attack.push_back(tmp);
        tmp.clear();
      } else {
        tmp += c;
      }
    }
  }
  /**********************STEP 3**********************/
  /*
    * The name of the output folder where the list of the neurons attacked will be saved is obtained.
  */
  if (rank == 0){
    std::ifstream file_config("/home/bmtk/v1_point/config.json");
    if (!file_config.is_open()) {
      throw std::runtime_error("Error opening the configuration file");
    }
    json config;
    file_config >> config;

    std::string output_dir = config["manifest"]["$OUTPUT_DIR"];
    std::string base_dir = ".";
    size_t pos = output_dir.find("$BASE_DIR");
    if (pos != std::string::npos) {
      output_dir.replace(pos, 9, base_dir);
    }
    std::string full_path = output_dir + "/list_id_attacked_FLO.csv";
    /*
      * Once the output file has been created, all the attacked IDs are inserted.
    */
    std::ofstream file_ids(full_path);  
    if (!file_ids.is_open()) {  
      std::cerr << "Error opening the file: " << full_path << std::endl;
      return;
    }

    for (const auto& str : id_neuron_attack) {  
      file_ids << str << "\n";  
    }

    file_ids.close(); 
  }
  /**********************STEP 4**********************/
  /*
    * The attack begins
  */
  // std::cout << "No. neurons FLO: " << id_neuron_attack.size() << std::endl; /**Optional**/
  std::ostringstream os;
  LOG( M_INFO, "SimulationManager::attack_based_FLO", os.str() );
  const size_t tid = kernel().vp_manager.get_thread_id();
  /*
    * The neurons are iterated to obtain their IDs, aiming to verify if they are on the list. 
    * If so, the cyberattack is executed by changing their membrane voltage to their corresponding threshold voltage.
  */
  for ( SparseNodeArray::const_iterator i = kernel().node_manager.get_local_nodes( tid ).begin();
    i != kernel().node_manager.get_local_nodes( tid ).end(); ++i )
  {
    Node* node = i->get_node();
    DictionaryDatum dict = kernel().node_manager.get_status(node->get_node_id());
    Dictionary dict_consulting = *dict;

    auto it = std::find(id_neuron_attack.begin(), id_neuron_attack.end(), std::to_string(node->get_node_id()));
    
    if(it != id_neuron_attack.end()){
      dict_consulting["V_m"] = dict_consulting["V_th"];
      node->set_status_base(dict_consulting);
    }
  }
}
/***********************************************************************/
/*
  * Function focused on execution the cyberattack JAM in the following attacks: JAM-JAM, JAM-FLO, and FLO-JAM.
*/
void
nest::SimulationManager::attack_based_JAM(std::string source_layer, std::string source_behaviour)
{
  /**********************STEP 1**********************/
  /*
    * As in the previous attack function, there was a need to implement a specific code to enable multiple threading. 
    * As well as the find_connection_ is used to create the neuron IDs list with specific parameters.
  */
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::string> id_neuron_attack;
  if (rank == 0) {
    id_neuron_attack = find_connection_(source_layer, source_behaviour);
    /*
      * Analysis of how many neurons are targeted by the attack. If `n_neurons` is 100, the cyberattack is applied 
      * to all selected neurons (100%).
    */
    std::random_device rd;
    std::mt19937 g(rd());
    int n = std::stoi(CONN_JAM_dict["n_neurons"]);
    if (n > 0 && n <= 100 && !id_neuron_attack.empty()) {
      std::shuffle(id_neuron_attack.begin(), id_neuron_attack.end(), g);
      size_t total = id_neuron_attack.size();
      size_t new_size = (total * n) / 100; 
      id_neuron_attack.resize(new_size);
    }   
  }
  /**********************STEP 2**********************/
  /*
    * Broadcast the list of attacked neuron IDs from rank 0 to all MPI processes.
    * Check the previous function to see a more detailed explanation. 
  */
  std::vector<char> flat_data;
  size_t num_ids = 0;

  if (rank == 0) {
    for (const auto& s : id_neuron_attack) {
      flat_data.insert(flat_data.end(), s.begin(), s.end());
      flat_data.push_back('\0');
    }
    num_ids = id_neuron_attack.size();
  }

  MPI_Bcast(&num_ids, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  size_t total_chars = flat_data.size();
  MPI_Bcast(&total_chars, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    flat_data.resize(total_chars);
  }

  MPI_Bcast(flat_data.data(), total_chars, MPI_CHAR, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    id_neuron_attack.clear();
    std::string tmp;
    for (char c : flat_data) {
      if (c == '\0') {
        id_neuron_attack.push_back(tmp);
        tmp.clear();
      } else {
        tmp += c;
      }
    }
  }
  /**********************STEP 3**********************/
  /*
    * The name of the output folder where the list of the neurons attacked will be saved is obtained.
  */
  if (rank == 0){
    std::ifstream file_config("/home/bmtk/v1_point/config.json");
    if (!file_config.is_open()) {
      throw std::runtime_error("Error opening the configuration file");
    }
    json config;
    file_config >> config;

    std::string output_dir = config["manifest"]["$OUTPUT_DIR"];
    std::string base_dir = ".";
    size_t pos = output_dir.find("$BASE_DIR");
    if (pos != std::string::npos) {
      output_dir.replace(pos, 9, base_dir);
    }
    std::string full_path = output_dir + "/list_id_attacked_JAM.csv";
    /*
      * Once the output file has been created, all the attacked IDs are inserted.
    */
    std::ofstream file_ids(full_path);  
    if (!file_ids.is_open()) {  
      std::cerr << "Error opening the file: " << full_path << std::endl;
      return;
    }

    for (const auto& str : id_neuron_attack) {  
      file_ids << str << "\n";  
    }

    file_ids.close(); 
  }
  /**********************STEP 4**********************/
  /*
    * The attack begins
  */
  // std::cout << ""No. neurons JAM: " << id_neuron_attack.size() << std::endl; /**Optional**/
  std::ostringstream os;
  LOG( M_INFO, "SimulationManager::attack_based_JAM", os.str() );
  const size_t tid = kernel().vp_manager.get_thread_id();
  /*
    * The neurons are iterated to obtain their IDs, aiming to verify if they are on the list. 
    * If so, the cyberattack is executed by changing their membrane voltage to their corresponding reset voltage.
  */
  for ( SparseNodeArray::const_iterator i = kernel().node_manager.get_local_nodes( tid ).begin();
    i != kernel().node_manager.get_local_nodes( tid ).end(); ++i )
  {
    Node* node = i->get_node();
    DictionaryDatum dict = kernel().node_manager.get_status(node->get_node_id());
    Dictionary dict_consulting = *dict;

    auto it = std::find(id_neuron_attack.begin(), id_neuron_attack.end(), std::to_string(node->get_node_id()));
    
    if(it != id_neuron_attack.end()){
      dict_consulting["V_m"] = dict_consulting["V_reset"];
      node->set_status_base(dict_consulting);
    }
  }
}
/***********************************************************************/

void
nest::SimulationManager::call_update_()
{
  assert( kernel().is_initialized() and not inconsistent_state_ );

  std::ostringstream os;
  double t_sim = to_do_ * Time::get_resolution().get_ms();

  size_t num_active_nodes = kernel().node_manager.get_num_active_nodes();
  os << "Number of local nodes: " << num_active_nodes << std::endl;
  os << "Simulation time (ms): " << t_sim;

#ifdef _OPENMP
  os << std::endl << "Number of OpenMP threads: " << kernel().vp_manager.get_num_threads();
#else
  os << std::endl << "Not using OpenMP";
#endif

#ifdef HAVE_MPI
  os << std::endl << "Number of MPI processes: " << kernel().mpi_manager.get_num_processes();
#else
  os << std::endl << "Not using MPI";
#endif

  LOG( M_INFO, "SimulationManager::start_updating_", os.str() );


  if ( to_do_ == 0 )
  {
    return;
  }

  if ( print_time_ )
  {
    // TODO: Remove direct output
    std::cout << std::endl;
    print_progress_();
  }

  simulating_ = true;
  simulated_ = true;

  update_();

  simulating_ = false;

  if ( print_time_ )
  {
    std::cout << std::endl;
  }

  kernel().mpi_manager.synchronize();

  LOG( M_INFO, "SimulationManager::run", "Simulation finished." );
}

void
nest::SimulationManager::update_connection_infrastructure( const size_t tid )
{
#pragma omp barrier
  if ( tid == 0 )
  {
    sw_communicate_prepare_.start();
  }

  kernel().connection_manager.restructure_connection_tables( tid );
  kernel().connection_manager.sort_connections( tid );
  kernel().connection_manager.collect_compressed_spike_data( tid );

#pragma omp barrier // wait for all threads to finish sorting

#pragma omp single
  {
    kernel().connection_manager.compute_target_data_buffer_size();
    kernel().event_delivery_manager.resize_send_recv_buffers_target_data();

    // check whether primary and secondary connections exists on any
    // compute node
    kernel().connection_manager.sync_has_primary_connections();
    kernel().connection_manager.check_secondary_connections_exist();
  }

  if ( kernel().connection_manager.secondary_connections_exist() )
  {
#pragma omp barrier
    kernel().connection_manager.compute_compressed_secondary_recv_buffer_positions( tid );
#pragma omp barrier
#pragma omp single
    {
      kernel().mpi_manager.communicate_recv_counts_secondary_events();
      kernel().event_delivery_manager.configure_secondary_buffers();
    }
  }

#ifdef TIMER_DETAILED
  if ( tid == 0 )
  {
    sw_gather_target_data_.start();
  }
#endif

  // communicate connection information from postsynaptic to
  // presynaptic side
  if ( kernel().connection_manager.use_compressed_spikes() )
  {
#pragma omp barrier
#pragma omp single
    {
      kernel().connection_manager.initialize_iteration_state(); // could possibly be combined with s'th above
    }
    kernel().event_delivery_manager.gather_target_data_compressed( tid );
  }
  else
  {
    kernel().event_delivery_manager.gather_target_data( tid );
  }

#ifdef TIMER_DETAILED
#pragma omp barrier
  if ( tid == 0 )
  {
    sw_gather_target_data_.stop();
  }
#endif

  if ( kernel().connection_manager.secondary_connections_exist() )
  {
    kernel().connection_manager.compress_secondary_send_buffer_pos( tid );
  }

#pragma omp barrier
#pragma omp single
  {
    kernel().connection_manager.clear_compressed_spike_data_map();
    kernel().node_manager.set_have_nodes_changed( false );
    kernel().connection_manager.unset_connections_have_changed();
  }

#pragma omp barrier
  if ( tid == 0 )
  {
    sw_communicate_prepare_.stop();
  }
}

bool
nest::SimulationManager::wfr_update_( Node* n )
{
  return ( n->wfr_update( clock_, from_step_, to_step_ ) );
}

void
nest::SimulationManager::update_()
{
  // to store done values of the different threads
  std::vector< bool > done;
  bool done_all = true;
  long old_to_step;

  double start_current_update = sw_simulate_.elapsed();
  bool update_time_limit_exceeded = false;

  std::vector< std::shared_ptr< WrappedThreadException > > exceptions_raised( kernel().vp_manager.get_num_threads() );

// parallel section begins
#pragma omp parallel
  {
    const size_t tid = kernel().vp_manager.get_thread_id();

    // We update in a parallel region. Therefore, we need to catch
    // exceptions here and then handle them after the parallel region.
    try
    {
      do
      {
        if ( print_time_ )
        {
          gettimeofday( &t_slice_begin_, nullptr );
        }

        // Do not deliver events at beginning of first slice, nothing can be there yet
        // and invalid markers have not been properly set in send buffers.
        if ( slice_ > 0 and from_step_ == 0 )
        {
          // Deliver secondary events before primary events
          //
          // Delivering secondary events ahead of primary events ensures that LearningSignalConnectionEvents
          // reach target neurons before spikes are propagated through eprop synapses.
          // This sequence safeguards the gradient computation from missing critical information
          // from the time step preceding the arrival of the spike triggering the weight update.
          if ( kernel().connection_manager.secondary_connections_exist() )
          {
#ifdef TIMER_DETAILED
            if ( tid == 0 )
            {
              sw_deliver_secondary_data_.start();
            }
#endif
            kernel().event_delivery_manager.deliver_secondary_events( tid, false );
#ifdef TIMER_DETAILED
            if ( tid == 0 )
            {
              sw_deliver_secondary_data_.stop();
            }
#endif
          }

          if ( kernel().connection_manager.has_primary_connections() )
          {
#ifdef TIMER_DETAILED
            if ( tid == 0 )
            {
              sw_deliver_spike_data_.start();
            }
#endif
            // Deliver spikes from receive buffer to ring buffers.
            kernel().event_delivery_manager.deliver_events( tid );

#ifdef TIMER_DETAILED
            if ( tid == 0 )
            {
              sw_deliver_spike_data_.stop();
            }
#endif
          }

#ifdef HAVE_MUSIC
// advance the time of music by one step (min_delay * h) must
// be done after deliver_events_() since it calls
// music_event_out_proxy::handle(), which hands the spikes over to
// MUSIC *before* MUSIC time is advanced

// wait until all threads are done -> synchronize
#pragma omp barrier
// the following block is executed by the master thread only
// the other threads are enforced to wait at the end of the block
#pragma omp master
          {
            // advance the time of music by one step (min_delay * h) must
            // be done after deliver_events_() since it calls
            // music_event_out_proxy::handle(), which hands the spikes over to
            // MUSIC *before* MUSIC time is advanced
            if ( slice_ > 0 )
            {
              kernel().music_manager.advance_music_time();
            }

            // the following could be made thread-safe
            kernel().music_manager.update_music_event_handlers( clock_, from_step_, to_step_ );
          }
// end of master section, all threads have to synchronize at this point
#pragma omp barrier
#endif
        } // if from_step == 0

        // preliminary update of nodes that use waveform relaxtion, only
        // necessary if secondary connections exist and any node uses
        // wfr
        if ( kernel().connection_manager.secondary_connections_exist() and kernel().node_manager.wfr_is_used() )
        {
#pragma omp single
          {
            // if the end of the simulation is in the middle
            // of a min_delay_ step, we need to make a complete
            // step in the wfr_update and only do
            // the partial step in the final update
            // needs to be done in omp single since to_step_ is a scheduler
            // variable
            old_to_step = to_step_;
            if ( to_step_ < kernel().connection_manager.get_min_delay() )
            {
              to_step_ = kernel().connection_manager.get_min_delay();
            }
          }

          bool max_iterations_reached = true;
          const std::vector< Node* >& thread_local_wfr_nodes = kernel().node_manager.get_wfr_nodes_on_thread( tid );
          for ( long n = 0; n < wfr_max_iterations_; ++n )
          {
            bool done_p = true;

            // this loop may be empty for those threads
            // that do not have any nodes requiring wfr_update
            for ( std::vector< Node* >::const_iterator i = thread_local_wfr_nodes.begin();
                  i != thread_local_wfr_nodes.end();
                  ++i )
            {
              done_p = wfr_update_( *i ) and done_p;
            }

// add done value of thread p to done vector
#pragma omp critical
            done.push_back( done_p );
// parallel section ends, wait until all threads are done -> synchronize
#pragma omp barrier

// the following block is executed by a single thread
// the other threads wait at the end of the block
#pragma omp single
            {
              // check whether all threads are done
              for ( size_t i = 0; i < done.size(); ++i )
              {
                done_all = done[ i ] and done_all;
              }

              // gather SecondaryEvents (e.g. GapJunctionEvents)
              kernel().event_delivery_manager.gather_secondary_events( done_all );

              // reset done and done_all
              //(needs to be in the single threaded part)
              done_all = true;
              done.clear();
            }

            // deliver SecondaryEvents generated during wfr_update
            // returns the done value over all threads
            done_p = kernel().event_delivery_manager.deliver_secondary_events( tid, true );

            if ( done_p )
            {
              max_iterations_reached = false;
              break;
            }
          } // of for (wfr_max_iterations) ...

#pragma omp single
          {
            to_step_ = old_to_step;
            if ( max_iterations_reached )
            {
              std::string msg = String::compose( "Maximum number of iterations reached at interval %1-%2 ms",
                clock_.get_ms(),
                clock_.get_ms() + to_step_ * Time::get_resolution().get_ms() );
              LOG( M_WARNING, "SimulationManager::wfr_update", msg );
            }
          }

        } // of if(wfr_is_used)
          // end of preliminary update

        if ( kernel().sp_manager.is_structural_plasticity_enabled()
          and ( std::fmod( Time( Time::step( clock_.get_steps() + from_step_ ) ).get_ms(),
                  kernel().sp_manager.get_structural_plasticity_update_interval() )
            == 0 ) )
        {
#pragma omp barrier
          for ( SparseNodeArray::const_iterator i = kernel().node_manager.get_local_nodes( tid ).begin();
                i != kernel().node_manager.get_local_nodes( tid ).end();
                ++i )
          {
            Node* node = i->get_node();
            node->update_synaptic_elements( Time( Time::step( clock_.get_steps() + from_step_ ) ).get_ms() );
          }
#pragma omp barrier
#pragma omp single
          {
            kernel().sp_manager.update_structural_plasticity();
          }
          // Remove 10% of the vacant elements
          for ( SparseNodeArray::const_iterator i = kernel().node_manager.get_local_nodes( tid ).begin();
                i != kernel().node_manager.get_local_nodes( tid ).end();
                ++i )
          {
            Node* node = i->get_node();
            node->decay_synaptic_elements_vacant();
          }

          // after structural plasticity has created and deleted
          // connections, update the connection infrastructure; implies
          // complete removal of presynaptic part and reconstruction
          // from postsynaptic data
          update_connection_infrastructure( tid );

        } // of structural plasticity


#ifdef TIMER_DETAILED
#pragma omp barrier
        if ( tid == 0 )
        {
          sw_update_.start();
        }
#endif
        const SparseNodeArray& thread_local_nodes = kernel().node_manager.get_local_nodes( tid );

        for ( SparseNodeArray::const_iterator n = thread_local_nodes.begin(); n != thread_local_nodes.end(); ++n )
        {
          Node* node = n->get_node();
          if ( not( node )->is_frozen() )
          {
            ( node )->update( clock_, from_step_, to_step_ );
          }
        }

// parallel section ends, wait until all threads are done -> synchronize
#pragma omp barrier

#ifdef TIMER_DETAILED
        if ( tid == 0 )
        {
          sw_update_.stop();
        }
#endif

        // the following block is executed by the master thread only
// the other threads are enforced to wait at the end of the block
#pragma omp master
        {
          // gather and deliver only at end of slice, i.e., end of min_delay step
          if ( to_step_ == kernel().connection_manager.get_min_delay() )
          {
            if ( kernel().connection_manager.has_primary_connections() )
            {
#ifdef TIMER_DETAILED
              sw_gather_spike_data_.start();
#endif

              kernel().event_delivery_manager.gather_spike_data();
#ifdef TIMER_DETAILED
              sw_gather_spike_data_.stop();
#endif
            }
            if ( kernel().connection_manager.secondary_connections_exist() )
            {
#ifdef TIMER_DETAILED
              sw_gather_secondary_data_.start();
#endif
              kernel().event_delivery_manager.gather_secondary_events( true );
            }
#ifdef TIMER_DETAILED
            sw_gather_secondary_data_.stop();
#endif
          }


          advance_time_();

          if ( print_time_ )
          {
            gettimeofday( &t_slice_end_, nullptr );
            print_progress_();
          }

          // We cannot throw exception inside master, would not get caught.
          const double end_current_update = sw_simulate_.elapsed();
          const double update_time = end_current_update - start_current_update;
          update_time_limit_exceeded = update_time > update_time_limit_;
          min_update_time_ = std::min( min_update_time_, update_time );
          max_update_time_ = std::max( max_update_time_, update_time );
          start_current_update = end_current_update;
          
          /*
            * During the simulation, NEST determines which attack is selected and the instant/interval when it must be applied.
            * It first checks whether the first attack is FLO or JAM. Then, it inspects the provided instants/intervals:
            * If more than one value is defined, the attack is executed consecutively, resulting in FLO–FLO or JAM–JAM.
          */
          if(type_attack["attack"] == "CONN_FLO")
          {
            std::stringstream ss(CONN_FLO_dict["instants_attack"]);
            std::string token;
            char delimiter = ',';

            while(std::getline(ss, token, delimiter))
            {
              if(clock_.get_ms() == std::stoi(token))
              {
                attack_based_FLO(CONN_FLO_dict["source_layer"], CONN_FLO_dict["source_behaviour"]);
              }
            }
          }else if(type_attack["attack"] == "CONN_JAM")
          {
            std::stringstream ss_end(CONN_JAM_dict["end_attack"]);
            std::stringstream ss_init(CONN_JAM_dict["init_attack"]);
            std::string token_init;
            std::string token_end;
            char delimiter = ',';
            while((std::getline(ss_init, token_init, delimiter)) && (std::getline(ss_end, token_end, delimiter)))
            {
              if (clock_.get_ms() >= std::stoi(token_init) && clock_.get_ms() <= std::stoi(token_end))
              {
                attack_based_JAM(CONN_JAM_dict["source_layer"], CONN_JAM_dict["source_behaviour"]);
              }
            }
          }
          /*
            * In this case, after selecting the first attack and confirming that only one instant/interval is provided, 
            * the code checks whether a second attack is defined, which differs from the first, resulting in FLO–JAM or JAM–FLO.   
          */
          if(type_attack["attack2"] == "CONN_FLO")
          {
            int init_instant = std::stoi(CONN_FLO_dict["instants_attack"]);

            if(clock_.get_ms() == init_instant)
            {
              attack_based_FLO(CONN_FLO_dict["source_layer"], CONN_FLO_dict["source_behaviour"]);
            }
          }else if(type_attack["attack2"] == "CONN_JAM")
          {
            int init_instant = std::stoi(CONN_JAM_dict["init_attack"]);
            int end_instant = std::stoi(CONN_JAM_dict["end_attack"]);
            
            if (clock_.get_ms() >= init_instant && clock_.get_ms() <= end_instant)
            {
              attack_based_JAM(CONN_JAM_dict["source_layer"], CONN_JAM_dict["source_behaviour"]);
            }
          }
        }
// end of master section, all threads have to synchronize at this point
#pragma omp barrier

        // if block to avoid omp barrier if SIONLIB is not used
#ifdef HAVE_SIONLIB
        kernel().io_manager.post_step_hook();
// enforce synchronization after post-step activities of the recording backends
#pragma omp barrier
#endif

        const double end_current_update = sw_simulate_.elapsed();
        if ( end_current_update - start_current_update > update_time_limit_ )
        {
          LOG( M_ERROR, "SimulationManager::update", "Update time limit exceeded." );
          throw KernelException();
        }
        start_current_update = end_current_update;

      } while ( to_do_ > 0 and not update_time_limit_exceeded and not exceptions_raised.at( tid ) );

      // End of the slice, we update the number of synaptic elements
      for ( SparseNodeArray::const_iterator i = kernel().node_manager.get_local_nodes( tid ).begin();
            i != kernel().node_manager.get_local_nodes( tid ).end();
            ++i )
      {
        Node* node = i->get_node();
        node->update_synaptic_elements( Time( Time::step( clock_.get_steps() + to_step_ ) ).get_ms() );
      }
    }
    catch ( std::exception& e )
    {
      // so throw the exception after parallel region
      exceptions_raised.at( tid ) = std::shared_ptr< WrappedThreadException >( new WrappedThreadException( e ) );
    }
  } // of omp parallel

  if ( update_time_limit_exceeded )
  {
    LOG( M_ERROR, "SimulationManager::update", "Update time limit exceeded." );
    throw KernelException();
  }

  // check if any exceptions have been raised
  for ( size_t tid = 0; tid < kernel().vp_manager.get_num_threads(); ++tid )
  {
    if ( exceptions_raised.at( tid ).get() )
    {
      simulating_ = false; // must mark this here, see #311
      inconsistent_state_ = true;
      throw WrappedThreadException( *( exceptions_raised.at( tid ) ) );
    }
  }
}

void
nest::SimulationManager::advance_time_()
{
  // time now advanced time by the duration of the previous step
  to_do_ -= to_step_ - from_step_;

  // advance clock, update modulos, slice counter only if slice completed
  if ( to_step_ == kernel().connection_manager.get_min_delay() )
  {
    clock_ += Time::step( kernel().connection_manager.get_min_delay() );
    ++slice_;
    kernel().event_delivery_manager.update_moduli();
    from_step_ = 0;
  }
  else
  {
    from_step_ = to_step_;
  }

  long end_sim = from_step_ + to_do_;

  if ( kernel().connection_manager.get_min_delay() < end_sim )
  {
    // update to end of time slice
    to_step_ = kernel().connection_manager.get_min_delay();
  }
  else
  {
    to_step_ = end_sim; // update to end of simulation time
  }

  assert( to_step_ - from_step_ <= kernel().connection_manager.get_min_delay() );
}

void
nest::SimulationManager::print_progress_()
{
  double rt_factor = 0.0;

  if ( t_slice_end_.tv_sec != 0 )
  {
    // usec
    long t_real_s = ( t_slice_end_.tv_sec - t_slice_begin_.tv_sec ) * 1e6;
    // usec
    t_real_ += t_real_s + ( t_slice_end_.tv_usec - t_slice_begin_.tv_usec );
    // ms
    double t_real_acc = ( t_real_ ) / 1000.;
    double t_sim_acc = ( to_do_total_ - to_do_ ) * Time::get_resolution().get_ms();
    // real-time factor = wall-clock time / model time
    rt_factor = t_real_acc / t_sim_acc;
  }

  int percentage = ( 100 - static_cast< int >( static_cast< double >( to_do_ ) / to_do_total_ * 100 ) );

  std::cout << "\r[ " << std::setw( 3 ) << std::right << percentage << "% ] "
            << "Model time: " << std::fixed << std::setprecision( 1 ) << clock_.get_ms() << " ms, "
            << "Real-time factor: " << std::setprecision( 4 ) << rt_factor
            << std::resetiosflags( std::ios_base::floatfield );
  std::flush( std::cout );
}

nest::Time const
nest::SimulationManager::get_previous_slice_origin() const
{
  return clock_ - Time::step( kernel().connection_manager.get_min_delay() );
}
