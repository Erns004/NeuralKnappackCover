import json
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_dense(n_neurons, activation, reg):
    # Crea capa densamente conectada
    return layers.Dense(n_neurons, activation=activation,
                        kernel_regularizer = reg,
                        bias_regularizer= reg)

def apply_concat_dense(network_inputs, prev_outputs, n_neurons, activation, reg):
    # concatena capas
    concat = layers.Concatenate()([network_inputs, prev_outputs])
    # Crea capa densa ¿y la conecta con las capas anteriores?
    return get_dense(n_neurons, activation, reg)(concat)

def get_nn_model(layers_width, reg):
    # capa que se ocupan como punto de entrada a una red
    all_inputs = layers.Input(shape=(layers_width[0],))
    # Crea primera capa, ¿y la conecta a la capa de input? 
    x = get_dense(layers_width[1], 'relu', reg)(all_inputs)
    for width in layers_width[2:-1]:
        x = apply_concat_dense(all_inputs, x, width, 'relu', reg)
    x = apply_concat_dense(all_inputs, x, layers_width[-1], None, reg)
    # Agrupa las capas en un objeto para entrenamiento y inferencia
    return tf.keras.Model(inputs=all_inputs, outputs=x)

def get_constant_width_model(cstar, width, reg):
    # input sequence: fin, pin, sin
    return get_nn_model([cstar+3, width, width, width, cstar+1], reg)

# Esto cambiarlo para generar instancias de nuestro problema
def get_random_knapsack_instance(cstar):
    remaining_cost = cstar
    costs = []
    while remaining_cost > 0:
        cost = np.random.randint(1, remaining_cost+1)
        remaining_cost -= cost
        costs.append(cost)
    costs = np.array(costs)
    np.random.shuffle(costs)
    sizes = np.random.random(len(costs))
    overload_factor = 1+np.random.random()
    sizes = sizes * overload_factor / sum(sizes)
    return costs, sizes

# Cambiar para que realize un paso en nuestro algoritmo de programción dinamica
def dp_step(f, cost, size):
    cstar = len(f)
    f_new = np.zeros(cstar)
    for c in range(cstar):
        use_item = 0
        if c >= cost:
            use_item = f[c-cost] + size
        f_new[c] = max(f[c], use_item)
    return f_new

def instance_to_training_data(costs, sizes, cstar):
    f_old = np.array([0] * (cstar + 1))
    training_data = []
    for cost, size in zip(costs, sizes):
        f_new = dp_step(f_old, cost, size)
        full_input = np.append(f_old, [cost, size])
        training_data.append((full_input, f_new))
        f_old = f_new
    return training_data

def prepare_dataset(cstar, batch_size, steps_per_epoch):
    def my_generator():
        while True:
            costs, sizes = get_random_knapsack_instance(cstar)
            current_data = instance_to_training_data(costs, sizes, cstar)
            for point in current_data:
                yield point
    # Crea los puntos de entrenamiento utilizando el generador ¿Crea puntos a medidad que se piden?
    dataset = tf.data.Dataset.from_generator(
        my_generator,
        (tf.float32, tf.float32),
        (tf.TensorShape([cstar+3]), tf.TensorShape([cstar+1])))
    # ¿Si ese el caso para que sirve mezclarlos? ¿Para generar un minimo de batch_size * steps_per_epoch?
    dataset = dataset.shuffle(batch_size * steps_per_epoch)
    dataset = dataset.batch(batch_size)
    return dataset

def evaluate_full_instance(cstar, model):
    costs, sizes = get_random_knapsack_instance(cstar)
    f_true = np.array([0] * (cstar + 1))
    f_model = np.array([0] * (cstar + 1))
    for cost, size in zip(costs, sizes):
        f_true = dp_step(f_true, cost, size)
        model_input = np.append(f_model, [cost, size])
        model_input = np.reshape(model_input, (1, len(model_input)))
        f_model = model(model_input)
    mse = tf.keras.losses.mse(f_true, f_model)
    return mse.numpy().item()

def compute_instance(cstar, costs, sizes):
    f_true = np.array([0] * (cstar + 1))
    for cost, size in zip(costs, sizes):
        f_true = dp_step(f_true, cost, size)
    print(f_true)
    for i in range(len(f_true)):
        if f_true[i] >=1:
            return i

def evaluate_full_instances(cstar, model, n_instances):
    mses = []
    for _ in range(n_instances):
        mses.append(evaluate_full_instance(cstar, model))
    return sum(mses)/len(mses)

def one_experiment(seed, cstar, width, reg, batch_size, steps_per_epoch, epochs, patience, verbose, eval_n):
    print('Current cstar = ' + str(cstar) + '. Current width = ' + str(width))
    tf.random.set_seed(seed)
    np.random.seed(seed)
    model = get_constant_width_model(cstar, width, reg)
    # Prepara el modelo para entrenamiento
    model.compile(loss = 'mse', optimizer = 'adam')
    dataset = prepare_dataset(cstar, batch_size, steps_per_epoch)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, verbose=verbose)]
    # entrena el modelo
    history = model.fit(dataset, epochs = epochs, steps_per_epoch = steps_per_epoch,
                verbose = verbose, callbacks=callbacks)
    # revisa el numero de epochs
    n_epochs = len(history.history['loss'])
    test_set = prepare_dataset(cstar, batch_size, steps_per_epoch)
    # evalua el modelo en el dataset de testeo
    loss = model.evaluate(test_set, verbose = verbose, steps = steps_per_epoch)
    mse = evaluate_full_instances(cstar, model, eval_n)
    print('MSE: ' + str(mse) + '; loss: ' + str(loss) + ' after ' + str(n_epochs) + ' epochs')
    return {
        "cstar": cstar,
        "width": width,
        "mse": mse,
        "loss": loss,
        "n_epochs": n_epochs
    }

def read_results(filename):
    f = open(filename)
    results = json.load(f)
    f.close()
    return results

if __name__ == '__main__':
    costs, sizes = get_random_knapsack_instance(10)
    print(costs, sizes)
    result = compute_instance(10, costs, sizes)
    print(result)
