
from world import World
from selection_funcs import Zone_selection, Load_custom
from obstacles import Wall, Load_wall
from custom_callbacks import InitLogs, LogWeights, LogWorldState, TimeGens



def main():
    world_shape = (50, 50)
    n_max_gen = 1_000
    n_logs = 10  

    death_func1 = Zone_selection("east_west", 0.1, world_shape)
    death_func2 = Zone_selection("circle", 0.1, world_shape)
    death_func3 = Zone_selection(("circle", "east_west"), (0.15, 0.06), world_shape)
    death_func4 = Zone_selection(("circle", "east_west", "north_south"), (0.15, 0.1, 0.1), world_shape)
    
    death_func_custom = Load_custom("death_masks", "newest")

    wall_mask1 = Wall(world_shape).mask

    wall_mask_custom = Load_wall("wall_masks", "newest").mask 

    # callbacks 
    init_logs = InitLogs()
    world_state_logging = LogWorldState('polynomial', n_max_gen, n_logs=n_logs, log_pos=False)
    weights_logging = LogWeights('polynomial', n_max_gen, n_logs=n_logs)
    time_gens = TimeGens(print_time_on_gen_end=True)

    world = World(world_shape=world_shape, n_population=150, n_steps=80, n_max_gen=n_max_gen,
                n_connections=12, create_logs=True,
                kill_enabled=True, wall_mask=wall_mask_custom, death_func=death_func_custom,
                no_spawn_in_zone=True,
                n_species=1, 
                callbacks=[init_logs, world_state_logging, time_gens])

    world.init_simulation() 

    world.start_simulation()


if __name__ == '__main__':
    main()       

        

        