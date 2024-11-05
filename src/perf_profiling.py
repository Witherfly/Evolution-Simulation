import cProfile
import pstats
import main


if __name__ == '__main__':

    with cProfile.Profile() as profile:
        print("HI")
        main.main()

    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats()
    results.dump_stats("perf_results.prof")
