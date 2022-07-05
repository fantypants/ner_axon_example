defmodule NerB do
  require Axon
  # From Example http://alexminnaar.com/2019/08/22/ner-rnns-tensorflow.html
  EXLA.set_as_nx_default([:tpu, :cuda, :rocm]) # My :host option doesnt work, jit_apply is undefined

  @sequence_length 100
  @batch_size 128
  @lstm 1024
  @embed_dimension 256
  @max_features 2000

  def build_model(vocab_count, label_count) do
    Axon.input({nil, @sequence_length, 1}, "input_chars")
    |> Axon.embedding(@max_features, @embed_dimension)
    |> Axon.dropout(rate: 0.01)
    |> Axon.conv(@batch_size, [strides: 3])
    |> Axon.conv(@batch_size, [strides: 3])
    |> Axon.global_max_pool
    |> Axon.dense(@batch_size, activation: :relu)
    |> Axon.dropout(rate: 0015)
    |> Axon.dense(label_count, activation: :sigmoid)

  end



  def transform_text(tokens, labels, label_count, vocab_count) do

      train_data =
        tokens
        |> Enum.chunk_every(@sequence_length, 1, :discard)
        |> Nx.tensor
        |> Nx.divide(vocab_count)
        |> Nx.reshape({:auto, @sequence_length, 1})
        |> IO.inspect
        |> Nx.to_batched_list(@batch_size)


    train_labels =
      labels
      |> Enum.chunk_every(@sequence_length, 1, :discard)
      |> Nx.tensor
    #  |> Nx.reshape({:auto, 1})
      #|> Nx.equal(Nx.iota{label_count})
      |> IO.inspect
      |> Nx.to_batched_list(@batch_size)

    {train_data, train_labels}
  end

  def log_metrics(state) do

    {:continue, state}
  end

  def rnn_loss(y_true, y_pred) do
    y_true
    |> Axon.Losses.categorical_cross_entropy(y_pred)
  #  |> Nx.sum()
  end

  def generate(model, params, init_seq, token_idx, char_idx) do
    IO.inspect token_idx
  init_seq =
    init_seq
    |> String.replace("\n", "")
    |> String.split(" ")
    |> Enum.reduce([], fn tk, acc ->

      tks =
      tk
      |> String.split("")
      |> Enum.map(fn t -> Map.get(token_idx, t) end)
      |> Enum.reject(fn t -> is_nil(t) end)


      acc ++ tks

    end)

    Enum.reduce(1..80, init_seq, fn _, seq ->
      init_seq =
        seq
      #  |> Enum.take(-@sequence_length)
        |> Nx.tensor()
      #  |> Nx.reshape({:auto, @sequence_length, 1})

      char =
        Axon.predict(model, params, init_seq)
        #|> IO.inspect
        |> Nx.mean()
        |> Nx.to_number()

      seq ++ [char]
    end)
    |> Enum.map(fn t ->
      #Map.get(char_idx, t)
      el = Enum.find(token_idx, fn {k, v} ->

        v == t

      end)

      if(!is_nil(el)) do
        elem(el, 0)
      end
    end)
    |> IO.inspect
  end


  def run do

    token_label_array = TextEntity.from_dataset("./data/train.txt")

    %{
      token_char_idx: token_char_idx,# Tokens characters as IDX [1,2,..,5]
      label_char_idx: label_char_idx,# Labels characters as IDX [1,2,..,5]
      token_idx: token_idx, # Token Dictionary Map
      label_idx: label_idx # Label Dictionary MAp
    } = Dictionary.from_token_label_stream(token_label_array)






    label_count = (label_idx |> Map.keys |> Enum.count)
    vocab_count = (token_idx |> Map.keys |> Enum.count)
    IO.puts "Characters: #{vocab_count} Labels: #{label_count}"
    {train_data, train_labels} =
      transform_text(token_char_idx, label_char_idx, label_count, vocab_count)

    model = build_model(vocab_count, label_count)
    IO.inspect(model)


    data = Stream.zip(train_data, train_labels)

    params =
      model
      |> Axon.Loop.trainer(:binary_cross_entropy, :adam)
      |> Axon.Loop.metric(:accuracy)
      |> Axon.Loop.handle(:started, &log_metrics/1, every: 1)
      |> Axon.Loop.handle(:iteration_completed, &log_metrics/1, every: 1)
      |> Axon.Loop.run(data, %{}, epochs: 3, iterations: 1)

      IO.inspect params
      init_seq = "B-ORG the big EU dog frog cat juice none zero that dawg fdsfs fsdfsdfsd fsdfsdf fdsf fdsf sdfsd fsd f sdf ds fsd"

      generate(model, params, init_seq, token_idx, label_idx)
      |> IO.inspect

  end
end
