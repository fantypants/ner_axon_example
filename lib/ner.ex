defmodule Ner do
  require Axon
  # From Example http://alexminnaar.com/2019/08/22/ner-rnns-tensorflow.html
  EXLA.set_as_nx_default([:tpu, :cuda, :rocm, :host]) # My :host option doesnt work, jit_apply is undefined

  @sequence_length 100
  @batch_size 128
  @lstm 1024
  @embed_dimension 256

  def build_model(vocab_count, label_count) do
    Axon.input({@batch_size, nil, 1}, "input_chars")
    #|> Axon.embedding(vocab_count, @embed_dimension)
    |> Axon.lstm(1024)
    |> then(fn {{cell, hidden}, out} ->
      out
    end)
    |> Axon.dropout(rate: 0.2)
    |> Axon.dense(label_count, activation: :softmax)

    #|> Axon.flatten()
    #|> Axon.dense(1, activation: :softmax)
    #

  end



  def transform_text(tokens, labels, label_count, vocab_count) do

      train_data =
        tokens
        |> Enum.chunk_every(@sequence_length, 1, :discard)
        |> Nx.tensor
      #  |> Nx.divide(vocab_count)
        |> Axon.Layers.embedding(Nx.iota{vocab_count, @embed_dimension})
        |> Nx.reshape({:auto, @sequence_length, 1})
        |> Nx.to_batched_list(@batch_size)


    train_labels =
      labels
      |> Enum.chunk_every(@sequence_length, 1, :discard)
      |> Nx.tensor
      |> Axon.Layers.embedding(Nx.iota{vocab_count, @embed_dimension})
      #|> Nx.reshape({:auto, 1})
      |> Nx.reshape({:auto, @sequence_length, 1})
      |> Nx.to_batched_list(@batch_size)

    {train_data, train_labels}
  end

  def log_metrics(state) do

    {:continue, state}
  end

  def rnn_loss(y_true, y_pred) do
    y_true
    |> Axon.Losses.categorical_cross_entropy(y_pred)
    |> Nx.sum()
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

    {train_data, train_labels} =
      transform_text(token_char_idx, label_char_idx, label_count, vocab_count)

      IO.inspect train_data

    model = build_model(vocab_count, label_count)
    IO.inspect(model)


    data = Stream.zip(train_data, train_labels)

    params =
      model
      |> Axon.Loop.trainer(:categorical_cross_entropy, :adam)
      |> Axon.Loop.metric(:accuracy)
      |> Axon.Loop.handle(:started, &log_metrics/1, every: 1)
      |> Axon.Loop.handle(:iteration_completed, &log_metrics/1, every: 1)
      |> Axon.Loop.run(data, %{}, epochs: 1, iterations: 2)

      IO.inspect params


  end
end
