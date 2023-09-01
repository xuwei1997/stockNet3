from train_class import StockPrediction
from Net import lstm_3_net_NOBN, lstm_3_net, bp_5_net
import pygad
import tensorflow
import pygad.kerasga
import tensorflow.keras
import sklearn


def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, keras_ga, model
    # print(data_inputs.shape)
    # print(data_outputs.shape)

    predictions = pygad.kerasga.predict(model=model,
                                        solution=solution,
                                        data=data_inputs)

    # mae = tensorflow.keras.losses.MeanAbsoluteError()
    # abs_error = mae(data_outputs, predictions).numpy() + 0.00000001

    mse = tensorflow.keras.losses.MeanSquaredError()
    abs_error = mse(data_outputs, predictions).numpy() + 0.00000001

    solution_fitness = 1.0 / abs_error
    # print(solution_fitness)

    return solution_fitness


def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


class psoLstmPrediction(StockPrediction):

    def train_main(self):
        self.data_preprocessing(windows=50)
        x_train, x_test, y_train, y_test, pre_close_train, pre_close_test = self.loda_data()

        global data_inputs, data_outputs, keras_ga, model

        data_inputs, data_outputs = x_train, y_train
        # model = lstm_3_net_NOBN(shape=(50, self.k_long))
        # model = lstm_3_net(shape=(50, self.k_long))
        model = bp_5_net(shape=(50, self.k_long))

        keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=20)

        # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
        # num_generations = 250  # Number of generations.
        num_generations = 100  # Number of generations.
        num_parents_mating = 5  # Number of solutions to be selected as parents in the mating pool.
        initial_population = keras_ga.population_weights  # Initial population of network weights

        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               initial_population=initial_population,
                               fitness_func=fitness_func,
                               on_generation=callback_generation)

        ga_instance.run()

        # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
        ga_instance.plot_fitness(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

        # Make prediction based on the best solution.
        predictions = pygad.kerasga.predict(model=model,
                                            solution=solution,
                                            data=x_test)
        # print("Predictions : \n", predictions)

        mae = tensorflow.keras.losses.MeanAbsoluteError()
        abs_error = mae(y_test, predictions).numpy()
        print("Absolute Error : ", abs_error)

        self.evaluates(y_test, predictions, pre_close_test)


if __name__ == '__main__':
    ticker_list = ['000998', '600598']
    # ticker_list = ['000998']
    for k in ticker_list:
        t = psoLstmPrediction(ticker=k)
        t.train_main()
