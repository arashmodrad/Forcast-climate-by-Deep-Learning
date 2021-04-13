import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import regularizers
from keras import backend as K

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

####################### Setup model parameters ######################
Run = '_Final'
Var_Name = 'T (degC)'
MAX_EPOCHS = 40

# Learn from previous 7 days
IN_STEPS = 24*6*7
# predict 1 day in advance
OUT_STEPS = 24*6

# Learn from previous 4 hour
CONV_WIDTH = 6*4 
# Learn from previous 7 days
CONV_WIDTH2 = 24*6*7

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


mini_batch_size = 16
cost_function = 'MSE'
optimizer = 'adam'
met = [coeff_determination]
eta = 0.0005

############################## Load Data ##########################

filename = 'Proccessed_Data/Data_Cleaned.csv' 
df = pd.read_csv(filename, index_col=None, header=0)

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.8)]
test_df = df[int(n*0.8):]

num_features = df.shape[1]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
plt.savefig('Range'+Run+'.png')

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    #self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
                     label_columns=[Var_Name])

w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                     label_columns=[Var_Name])

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window

# Stack three slices, the length of the total window:
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])


example_inputs, example_labels = w2.split_window(example_window)

w2.example = example_inputs, example_labels

def plot(self, model=None, save_name = 'Default', plot_col=Var_Name, max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(3, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [10 min]')
  plt.savefig(save_name+Run+'.png')

WindowGenerator.plot = plot


def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset


@property
def train(self):
  return self.make_dataset(self.train_df)

#@property
#def val(self):
#  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
#WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

w2.train.element_spec

for example_inputs, example_labels in w2.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')

single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['T (degC)'])
single_step_window

for example_inputs, example_labels in single_step_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')

  def plotgraph(epochs, acc, val_acc, save_name = 'Default'):
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(save_name+Run+'.png')


class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

baseline = Baseline(label_index=column_indices[Var_Name])

#baseline.compile(loss=tf.losses.MeanSquaredError(),
#                 metrics=[tf.metrics.MeanAbsoluteError()])

#val_performance = {}
performance = {}
#val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
#performance['Baseline'] = baseline.evaluate(single_step_window.test)


wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    label_columns=[Var_Name])

wide_window

linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])


def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.test,
                      callbacks=[early_stopping])
  return history

best_weights="cnn_weights_best.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(best_weights, monitor='val_acc', verbose=1, save_best_only=True,
                                                mode='max')
def compile_and_fit2(model, window, patience=2):
  if(optimizer=='adam'):
      optim = tf.keras.optimizers.Adam(lr=eta)
  else:
      optim = tf.keras.optimizers.SGD(lr=eta)
      
  model.compile(loss=cost_function,
                optimizer=optim,
                metrics = met)

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      batch_size=mini_batch_size,
                      validation_data=window.test,
                      callbacks=[checkpoint])
  return history

#history = compile_and_fit(linear, single_step_window)

#val_performance['Linear'] = linear.evaluate(single_step_window.val)
#performance['Linear'] = linear.evaluate(single_step_window.test)

class ResidualWrapper(tf.keras.Model):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def call(self, inputs, *args, **kwargs):
    delta = self.model(inputs, *args, **kwargs)

    # The prediction for each timestep is the input
    # from the previous time step plus the delta
    # calculated by the model.
    return inputs + delta



######################################## Multi-step models ##########################################

multi_window = WindowGenerator(input_width=IN_STEPS ,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

multi_window.plot(save_name='Pre')
multi_window

class MultiStepLastBaseline(tf.keras.Model):
  def call(self, inputs):
    return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

last_baseline = MultiStepLastBaseline()

multi_performance = {}

class RepeatBaseline(tf.keras.Model):
  def call(self, inputs):
    return inputs

repeat_baseline = RepeatBaseline()



######### Single-shot models ###############
# Linear

multi_linear_model = tf.keras.Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit2(multi_linear_model, multi_window)

IPython.display.clear_output()
#multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test)
multi_window.plot(multi_linear_model, save_name='Linear')
print(multi_linear_model.summary())

acc = history.history['coeff_determination']
val_acc = history.history['val_coeff_determination']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plotgraph(epochs, acc, val_acc, save_name = 'ACC_Linear_OVERfit')
plotgraph(epochs, loss, val_loss, save_name = 'LOSS_Linear_OVERfit')


# Dense1

multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation='relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit2(multi_dense_model, multi_window)

IPython.display.clear_output()
#multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
multi_performance['Dense1'] = multi_dense_model.evaluate(multi_window.test)
multi_window.plot(multi_dense_model, save_name='Dense1')
print(multi_dense_model.summary())

acc = history.history['coeff_determination']
val_acc = history.history['val_coeff_determination']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plotgraph(epochs, acc, val_acc, save_name = 'ACC_Dense1_OVERfit')
plotgraph(epochs, loss, val_loss, save_name = 'LOSS_Dense1_OVERfit')



# Dense2
multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(1024, activation='relu'),

    tf.keras.layers.Dense(512, activation='relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit2(multi_dense_model, multi_window)

IPython.display.clear_output()
#multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
multi_performance['Dense2'] = multi_dense_model.evaluate(multi_window.test)
multi_window.plot(multi_dense_model, save_name='Dense2')
print(multi_dense_model.summary())

acc = history.history['coeff_determination']
val_acc = history.history['val_coeff_determination']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plotgraph(epochs, acc, val_acc, save_name = 'ACC_Dense2_OVERfit')
plotgraph(epochs, loss, val_loss, save_name = 'LOSS_Dense2_OVERfit')

# CNN1


multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit2(multi_conv_model, multi_window)

IPython.display.clear_output()

#multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv1'] = multi_conv_model.evaluate(multi_window.test)
multi_window.plot(multi_conv_model, save_name='CNN1')
print(multi_conv_model.summary())


acc = history.history['coeff_determination']
val_acc = history.history['val_coeff_determination']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plotgraph(epochs, acc, val_acc, save_name = 'ACC_CNN1_OVERfit')
plotgraph(epochs, loss, val_loss, save_name = 'LOSS_CNN1_OVERfit')

# CNN2


multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH2:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH2)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit2(multi_conv_model, multi_window)

IPython.display.clear_output()

#multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv2'] = multi_conv_model.evaluate(multi_window.test)
multi_window.plot(multi_conv_model, save_name='CNN2')
print(multi_conv_model.summary())

acc = history.history['coeff_determination']
val_acc = history.history['val_coeff_determination']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plotgraph(epochs, acc, val_acc, save_name = 'ACC_CNN2_OVERfit')
plotgraph(epochs, loss, val_loss, save_name = 'LOSS_CNN2_OVERfit')

# RNN1

multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit2(multi_lstm_model, multi_window)

IPython.display.clear_output()

#multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['RNN1'] = multi_lstm_model.evaluate(multi_window.test)
multi_window.plot(multi_lstm_model, save_name='RNN1')
print(multi_lstm_model.summary())

acc = history.history['coeff_determination']
val_acc = history.history['val_coeff_determination']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plotgraph(epochs, acc, val_acc, save_name = 'ACC_RNN1_OVERfit')
plotgraph(epochs, loss, val_loss, save_name = 'LOSS_RNN1_OVERfit')

# RNN2


multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(units = 1024, activation='tanh', recurrent_activation='sigmoid',
                          use_bias=True, kernel_initializer='glorot_uniform',
                          recurrent_initializer='orthogonal',
                          bias_initializer='zeros', unit_forget_bias=True,
                          kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                          activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
                          bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=2,
                          return_sequences=False, return_state=False, go_backwards=False, stateful=False,
                          time_major=False, unroll=False
    ),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(units= OUT_STEPS*num_features, activation=None, use_bias=True,
                          kernel_initializer='glorot_uniform',
                          bias_initializer=tf.initializers.zeros, kernel_regularizer=None,
                          bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                          bias_constraint=None
    ),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit2(multi_lstm_model, multi_window)

IPython.display.clear_output()

#multi_val_performance['LSTM1'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['RNN2'] = multi_lstm_model.evaluate(multi_window.test)
multi_window.plot(multi_lstm_model, save_name='RNN2')
print(multi_lstm_model.summary())

acc = history.history['coeff_determination']
val_acc = history.history['val_coeff_determination']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plotgraph(epochs, acc, val_acc, save_name = 'ACC_RNN2_OVERfit')
plotgraph(epochs, loss, val_loss, save_name = 'LOSS_RNN2_OVERfit')


############################# Autoregressive model ###################################
# AR_LSTM1
class FeedBack(tf.keras.Model):
  def __init__(self, units, out_steps):
    super().__init__()
    self.out_steps = out_steps
    self.units = units
    self.lstm_cell = tf.keras.layers.LSTMCell(units)
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
    self.dense = tf.keras.layers.Dense(num_features)

feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)

def warmup(self, inputs):
  # inputs.shape => (batch, time, features)
  # x.shape => (batch, lstm_units)
  x, *state = self.lstm_rnn(inputs)

  # predictions.shape => (batch, features)
  prediction = self.dense(x)
  return prediction, state

FeedBack.warmup = warmup

prediction, state = feedback_model.warmup(multi_window.example[0])
prediction.shape

def call(self, inputs, training=None):
  # Use a TensorArray to capture dynamically unrolled outputs.
  predictions = []
  # Initialize the lstm state
  prediction, state = self.warmup(inputs)

  # Insert the first prediction
  predictions.append(prediction)

  # Run the rest of the prediction steps
  for n in range(1, self.out_steps):
    # Use the last prediction as input.
    x = prediction
    # Execute one lstm step.
    x, state = self.lstm_cell(x, states=state,
                              training=training)
    # Convert the lstm output to a prediction.
    prediction = self.dense(x)
    # Add the prediction to the output
    predictions.append(prediction)

  # predictions.shape => (time, batch, features)
  predictions = tf.stack(predictions)
  # predictions.shape => (batch, time, features)
  predictions = tf.transpose(predictions, [1, 0, 2])
  return predictions

FeedBack.call = call


history = compile_and_fit2(feedback_model, multi_window)

IPython.display.clear_output()

#multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
multi_performance['AR_LSTM1'] = feedback_model.evaluate(multi_window.test)
multi_window.plot(feedback_model, save_name='AR_LSTM1')
print(feedback_model.summary())


acc = history.history['coeff_determination']
val_acc = history.history['val_coeff_determination']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plotgraph(epochs, acc, val_acc, save_name = 'ACC_AR_LSTM1_OVERfit')
plotgraph(epochs, loss, val_loss, save_name = 'LOSS_AR_LSTM1_OVERfit')

# AR_LSRM2

class FeedBack(tf.keras.Model):
  def __init__(self, units, out_steps):
    super().__init__()
    self.out_steps = out_steps
    self.units = units
    self.lstm_cell = tf.keras.layers.LSTMCell(units, activation='swish', recurrent_activation='sigmoid', use_bias=True,
                                              kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                                              bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer= tf.keras.regularizers.l1(0.01),
                                              recurrent_regularizer= tf.keras.regularizers.l1(0.01), bias_regularizer=None, kernel_constraint=None,
                                              recurrent_constraint=None, bias_constraint=None, dropout=0.08,
                                              recurrent_dropout=0.1, implementation=2,
    )
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
    self.dense = tf.keras.layers.Dense(num_features)

feedback_model = FeedBack(units=240, out_steps=OUT_STEPS)

FeedBack.warmup = warmup

prediction, state = feedback_model.warmup(multi_window.example[0])
prediction.shape

#print('Output shape (batch, time, features): ', feedback_model(multi_window.example[0]).shape)
FeedBack.call = call

history = compile_and_fit2(feedback_model, multi_window)

IPython.display.clear_output()
print(feedback_model.summary())

#multi_val_performance['AR LSTM2'] = feedback_model.evaluate(multi_window.val)
multi_performance['AR_LSTM2'] = feedback_model.evaluate(multi_window.test)
multi_window.plot(feedback_model, save_name='AR_LSTM2')

acc = history.history['coeff_determination']
val_acc = history.history['val_coeff_determination']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plotgraph(epochs, acc, val_acc, save_name = 'ACC_AR_LSTM2_OVERfit')
plotgraph(epochs, loss, val_loss, save_name = 'LOSS_AR_LSTM2_OVERfit')



x = np.arange(len(multi_performance))
width = 0.3


metric_name = 'accuracy'
metric_index = multi_lstm_model.metrics_names.index('accuracy')
#val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

#plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(),
           rotation=45)
#plt.ylabel(f'MAE (average over all times and outputs)')
plt.ylabel(f'Accuracy')
_ = plt.legend()
plt.savefig('comp'+Run+'.png')

for name, value in multi_performance.items():
  print(f'{name:8s}: {value[1]:0.4f}')