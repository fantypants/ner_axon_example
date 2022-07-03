defmodule EntityExtraction do
  require Axon
  require Logger


  @sequence_length 128
  @epochs 10
  @learning_rate 0.001
  @loss :mean_absolute_error
  @dropout_rate 0.1

  @rnn_units 1024
  @embed_dim 256
  @batch_size 128

  @input_columns [
    "Token"
  ]

  def transform_data(text, split_count) do
    return_data =
      text
      |> Enum.chunk_every(split_count, 1, :discard)
      |> IO.inspect
      |> Nx.tensor()
      #|>
      ##|> IO.inspect
      |> Nx.to_batched_list(@batch_size)
      |> Enum.map(fn btch ->

        btch
        |> Nx.reshape({128, 1, split_count})


      end)

    return_data
  end

  def generate_dataset(tokens, entities) do



    tok_count = tokens |> Enum.count
    IO.puts "token Count: #{tok_count}"

    {char_count, _} =
      (tok_count / @batch_size)
      |> Float.to_string
      |> Integer.parse

    IO.puts "Char Count: #{char_count}"
    #data = Enum.zip(tokens, entities) |> IO.inspect

    ts = transform_data(tokens, char_count) |> IO.inspect
    ls = transform_data(entities, char_count) |> IO.inspect


    token_tensors = ts

  #    |> Nx.reshape({:auto, 128})

    #  |> IO.inspect
    #  |> Nx.reshape({@batch_size, :auto, 1})

  #  IO.puts "Tensor count #{Enum.count(token_tensors)}"

      #{batch_count, 128} = Nx.shape(token_tensors)

    #  IO.puts "Total Samples: #{batch_count}"

      #|> Enum.map(fn x -> Explorer.from_list(x) |> IO.inspect end)

      #|> Enum.reduce([], fn sq, acc ->

    #    IO.inspect sq
    #    acc

    #  end)
      # We don't want the last chunk since we don't have a prediction for it.
    #  |> Enum.drop(-1)
      #|> Nx.tensor()
    #  |> IO.inspect


    #  size = Nx.size(token_tensors) |> IO.inspect

    #  padding = Enum.map()
      #|> Nx.divide(@batch_size)
    #  |> Nx.pad(0, [{0, 1, 0}])
      #|> Nx.reshape({:auto, 1, @batch_size})
      #|> Nx.to_batched_list(@batch_size)


        # Enum.chunk(tokens, 128)
      #  |> Nx.tensor
        # |> Nx.reshape({22, 1, 128})
         #|> IO.inspect


         # Array -> 128 Arrays

            #


         target_tensors = ls
         #|> Enum.chunk(@batch_size)
        # |> Nx.tensor()
         #|> Nx.to_batched_list(char_count)
         #|> Nx.stack

        # |> IO.inspect
         #|> Nx.reshape({@batch_size, :auto, 1})


        # {tbatch_count, 128} = Nx.shape(target_tensors)

         #IO.puts "Total Samples: #{tbatch_count}"


     {token_tensors, target_tensors}

  end

  def run_example() do
    # split the data into test and train sets, each with inputs and targets
  #  {test_inputs, test_targets, train_inputs, train_targets} =
  #{test_inputs, train_inputs} =
    #{token_dict, entity_dict} =
      {gen_train_series_token_map, gen_train_series_entity_map} =
        TextEntity.gen_train_series(
        "./data/train.txt",
        input_columns: @input_columns
      )


      %{dictionary: token_dictionary, character_elements: gen_train_series_tokens} = gen_train_series_token_map

      %{dictionary: label_dictionary, character_elements: gen_train_series_entities} = gen_train_series_entity_map


      {token_series, target_series} = generate_dataset(gen_train_series_tokens, gen_train_series_entities)

      vocab_size = Map.keys(token_dictionary) |> Enum.count |> IO.inspect

      label_size = Map.keys(label_dictionary) |> Enum.count |> IO.inspect


    model = build_model(vocab_size, label_size)

      IO.inspect(model)


      #input = Nx.tensor(token_series)

      #kernels = Nx.tensor(target_series)

    #  Axon.Layers.embedding(token_series, target_series) |> IO.inspect

  train(model, token_series, target_series)

    # make some predictions
  #  test_inputs
    #|> Enum.zip(test_targets)
  #  |> Enum.each(fn {phrase_input, actual_index} ->
  #   predicted_index = predict(model, phrase_input)
  #    Logger.info("Actual: #{scalar(actual_index)}. Predicted: ##{predicted_index}")
  #  end)
  end

  defp build_model(vocab_size, label_size) do
    IO.puts "Vocab Size: #{vocab_size}, label_size: #{label_size}"
    Axon.input({nil, 1, 256}, "input_chars")
  #  |> Axon.Layers.embedding(vocab_size, 256)
    #|> IO.inspect
    |> Axon.lstm(1024)
    |> then(fn {_, out} -> IO.inspect out end)
  #  |> Axon.nx(fn t -> t[[0..-1//1, -1]] end)
    #|> Axon.dropout(rate: 0.2)
    |> Axon.dense(label_size, activation: :softmax)
  end

  def log_metrics(state) do

    IO.inspect state
    {:continue, state}
  end

  def train(model, inputs, targets) do
    IO.puts "Training Start"
  #  model =
      # Input is Root Layer
    #  Axon.Layers.embedding(t, z)
    #  |> IO.inspect
      #Axon.input({@batch_size, 1, Enum.count(inputs)}, "input")

    #  |> Axon.lstm(@rnn_units)
      #|> Axon.dense(Enum.count(targets))
      data = Stream.zip(inputs, targets)

    trained_params =
      model
      |> Axon.Loop.trainer(&rnn_loss/2, :adam)
      |> Axon.Loop.metric(:accuracy)
      |> Axon.Loop.handle(:iteration_completed, &log_metrics/1, every: 1)
      |> Axon.Loop.run(data, %{}, epochs: @epochs, iterations: 10)


      trained_params |> IO.inspect


  end

  def rnn_loss(y_true, y_pred) do
    y_true
    |> Axon.Losses.categorical_cross_entropy(y_pred)
    |> Nx.sum()
  end


  def train_old(inputs, targets) do
    model =
      Axon.input({1, Enum.count(@input_columns) + 1}, "input")
      |> Axon.dense(10)
      |> Axon.dropout(rate: @dropout_rate)
      |> Axon.dense(1)

    Logger.info(inspect(model))

    #optimizer = Axon.Optimizers.adamw(@learning_rate)

  #  %{params: trained_params} =
     #model
    #  |> Axon.Loop.trainer(@loss, optimizer)
    #  |> Axon.Loop.run(inputs, targets, epochs: @epochs, compiler: EXLA)


data = Stream.zip(inputs, targets)
      # trainer()
        trained_params =
          model
          |> Axon.Loop.trainer(:mean_squared_error, :adam)
          |> IO.inspect
        #  |> Axon.Loop.handle(:iteration_completed, &log_metrics(&1, :test), every: 50)
          |> Axon.Loop.run(data, %{}, epochs: @epochs, iterations: 1000) |> IO.inspect

#
  #  model
  #  |> Axon.Loop.trainer(@loss, optimizer)
  #  |> Axon.Loop.run(inputs, targets, epochs: @epochs, compiler: EXLA)





    {model, trained_params}
  end

  def predict({model, trained_params}, phrase_input) do
    model
    |> Axon.predict(trained_params, phrase_input)
    |> Nx.to_flat_list()
    |> List.first()
  end

  def scalar(tensor) do
    tensor |> Nx.to_flat_list() |> List.first()
  end
end
