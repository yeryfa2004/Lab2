import random
import numpy as np

from deap import base, creator, tools
LOW, UP = -3, 3
ETA = 20
LENGTH_CHROM = 3

POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.2
MAX_GENERATION = 50

def eval_func(individual):
    x, y, z = individual
    return 1/(1 + (x-2) ** 2 + (y+1) ** 2 + (z-1) ** 2),


def random_point(a, b):
    return [random.uniform(a, b), random.uniform(a, b), random.uniform(a, b)]


def create_toolbox():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("randomPoint", random_point, LOW, UP)
    toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomPoint)
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

    toolbox.register('evaluate', eval_func)

    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('mate', tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA)
    toolbox.register('mutate', tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0 / LENGTH_CHROM)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('min', np.min)
    stats.register('avg', np.mean)

    return toolbox


if __name__ == "__main__":
    toolbox = create_toolbox()
    random.seed(7)
    population = toolbox.populationCreator(n=500)
    num_generations = 100
    print("\nStarting the evolution process")
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    print('\nEvaluted', len(population), 'individuals')
    for g in range(num_generations):
        print('\n===== Generation ', g)
        offspring = toolbox.select(population, len(population))

        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)

                del child1.fitness.values
                del child1.fitness.values

        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.values]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print('Evaluated ', len(invalid_ind), 'individuals')
        population[:] = offspring
        fits = [ind.fitness.values[0] for ind in population]

        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        print('Min =', min(fits), ' Max =', max(fits))
        print('Average = ', round(mean, 2), ' Standard deviation = ', round(std, 2))
    print('\n===== End of evolution')
    best_ind = tools.selBest(population, 1)[0]
    print('\nBest individual', best_ind)
    print('\nNumber of ones', sum(best_ind))
