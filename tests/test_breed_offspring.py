import pytest 

from src.breed_offspring import crossover, bit_flip_mutation, create_offspring
from src.dot import Dot 

from unittest import mock


params = [(10, 20), (10, 30), (3, 10), (2, 10), (4, 5), (2, 7)]

@pytest.mark.parametrize('n_survivors, n_population', params)
def test_crossover(n_population, n_survivors):


    dot_objects = [Dot(i, '') for i in range(n_survivors)]

    def crossover_func(parents):

        genome = str(parents[0].id).zfill(4) + str(parents[1].id).zfill(4)

        return (Dot(0, genome), Dot(1, genome))
    
    offspring = crossover(dot_objects, crossover_func, n_population)

    assert len(offspring) == n_population

@mock.patch("src.breed_offspring.np.random.default_rng")
def test_bit_flip_mutation(mocked):

    genomes = ['00000', '00000', '00000', '11111', '11111', '11111', '00000']

    
    dot_objects = [Dot(i, genome) for i, genome in enumerate(genomes)]

    mocked.return_value.integers.side_effect =  [0, 2, 4, 0, 2, 4, 0, 1, 2, 3, 3]
    mocked.return_value.binomial.side_effect = [1, 1, 1, 1, 1, 1, 5] 

    mutated_dot_objects = bit_flip_mutation(dot_objects)


    assert mutated_dot_objects[0].genome == '10000'
    assert mutated_dot_objects[1].genome == '00100'
    assert mutated_dot_objects[2].genome == '00001'
    assert mutated_dot_objects[3].genome == '01111'
    assert mutated_dot_objects[4].genome == '11011'
    assert mutated_dot_objects[5].genome == '11110'
    assert mutated_dot_objects[6].genome == '11110'
    assert len(mutated_dot_objects) == len(genomes)

    mutated_dot_objects = bit_flip_mutation([])

    assert mutated_dot_objects == []

params = [(10, 50, 0.5), (0, 10, 0.5), (1, 10, 0.5), (3, 10, 0.5), 
          (10, 50, 0.1), (10, 50, 0.9), (10, 10, 0.3), (26, 150, 0.8)]

@pytest.mark.parametrize('n_parents, n_population, crossover_rate', params)
def test_create_offspring(n_parents, n_population, crossover_rate):

    parents = [Dot(i, '00000') for i in range(n_parents)]
    dot_objects = [Dot(i, '11111') for i in range(n_population - n_parents)] + parents

    offspring = create_offspring(parents, dot_objects, n_population, crossover_rate=crossover_rate)

    assert len(offspring) == n_population

    unique_ids = set()
    for dot in offspring:
        assert dot.id not in unique_ids
        unique_ids.add(dot.id)




