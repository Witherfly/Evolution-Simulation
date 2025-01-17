
from world import World
from selection_funcs import Zone_selection, Load_custom_death
from obstacles import Wall, Load_custom_wall
from custom_callbacks import InitLogs, LogWeights, LogWorldState, TimeGens, TimeGensLogger
from dot import DotGeneless, DotGenetic
from breed_offspring import weights_row_crossover, one_point_crossover, weights_mutation
from live_plot import StatCollectorMultiSpecies, StatCollector



def main():
    world_shape = (50, 50)
    n_max_gen = 1_000
    n_logs = 20
    n_species = 1
    dot_type = DotGeneless
    crossover_func = weights_row_crossover
    mutation_func = weights_mutation
    kill_enabled = True
    stats_collector = StatCollectorMultiSpecies(n_species, kill_enabled=kill_enabled)

    death_func1 = Zone_selection("east_west", 0.05, world_shape)
    death_func2 = Zone_selection("circle", 0.1, world_shape)
    death_func3 = Zone_selection(("circle", "east_west"), (0.15, 0.05), world_shape)
    death_func4 = Zone_selection(("circle", "east_west", "north_south"), (0.15, 0.1, 0.1), world_shape)
    
    death_func_custom = Load_custom_death("newest", world_shape)

    wall_mask1 = Wall(world_shape).mask

    wall_mask_custom = Load_custom_wall("newest", world_shape).mask 

    # callbacks 
    init_logs = InitLogs()
    world_state_logging = LogWorldState('polynomial', n_max_gen, n_logs=n_logs, log_pos=False)
    weights_logging = LogWeights('polynomial', n_max_gen, n_logs=n_logs)
    time_gens = TimeGens(print_time_on_gen_end=True)
    time_gens_logger = TimeGensLogger(logging_intervall=10)

    world = World(world_shape=world_shape, n_population=200, n_steps=70, n_max_gen=n_max_gen,
                  dot_type=dot_type, crossover_func=crossover_func, mutation_func=mutation_func,
                  n_neurons_hidden_layer=(8, 6),
                n_connections=14, create_logs=True, live_plotting=True,
                kill_enabled=kill_enabled, 
                wall_mask=wall_mask_custom, 
                death_func=death_func_custom,
                no_spawn_in_zone=True,
                n_species=n_species, 
                plotting_stat_collector=stats_collector,
                species_obs=False,
                dot_density_obs=True,
                trans_species_killing='no_restriction',
                callbacks=[init_logs, world_state_logging, time_gens, time_gens_logger])

    world.init_simulation() 

    world.start_simulation()


if __name__ == '__main__':
    main()       

        

        