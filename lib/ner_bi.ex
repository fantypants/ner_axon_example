defmodule BiNer do
  require Axon
  # From Example http://alexminnaar.com/2019/08/22/ner-rnns-tensorflow.html
  EXLA.set_as_nx_default([:tpu, :cuda, :rocl, :host]) # My :host option doesnt work, jit_apply is undefined

  @sequence_length 100
  @batch_size 128
  @lstm 1024
  @embed_dimension 256

  # seqlen
  # word_count


  def train(model, data) do

    model
    |> Axon.Loop.trainer(&rnn_loss/2, Axon.Optimizers.adam(0.001))
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(data, %{}, epochs: 10, iterations: 3)

  end

  def log_metrics(state) do
    IO.inspect state
    {:continue, state}
  end

  def rnn_loss(y_true, y_pred) do
    y_true
    |> Axon.Losses.categorical_cross_entropy(y_pred, [sparse: true])
    |> Nx.sum()
  end

  def build_model(word_count, label_count, max_length) do

      Axon.input({nil, max_length, 1}, "inputs")
       |> Axon.embedding(word_count, max_length)
       |> Axon.nx(fn t ->
         t[[0..-1//1, -1]]
     end)
       |> Axon.spatial_dropout(rate: 0.1)
       |> Axon.lstm(max_length)
       |> then(fn {{new_cell, new_hidden}, out} ->
         out
       end)
      |> Axon.dropout(rate: 0.2)
      |> Axon.dense(label_count, activation: :softmax)

  end

  def padded_batches(word_label_array, max_len, word_count, label_count) do
      word_label_array
      |> Enum.map(fn [token, labels] ->

        token_length = token |> Enum.count
        label_length = labels |> Enum.count


        post_pad_length = max_len - token_length


        n = Range.new(0, post_pad_length)
        padded_range = Enum.map(n, fn x -> "" end)
        label_padded_range = Enum.map(n, fn x -> "O" end)
        [token ++ padded_range, labels ++ label_padded_range]
      end)
  end

  def padded_transform(char_label_array, char_to_idx, label_to_idx, max_length) do

    train_data =
      char_label_array
      |> Enum.map(fn [token, labels] ->

        token
        |> Enum.map(&Map.fetch!(char_to_idx, &1))
        |> Nx.tensor



      end)
      |> Nx.stack
      |> Nx.reshape({:auto, max_length+1, 1})
      |> Nx.to_batched_list(@batch_size)


      train_labels =
        char_label_array
        |> Enum.map(fn [token, labels] ->

          labels
          |> Enum.map(&Map.fetch!(label_to_idx, &1))
          |> Nx.tensor



        end)
        |> Nx.stack
        |> Nx.reshape({:auto, max_length+1, 1})
        |> Nx.to_batched_list(@batch_size)



        {train_data, train_labels}
  end

  def transform_text({words, labels}, {word_count, label_count}) do

    train_data =
      words
      |> Enum.chunk_every(@sequence_length, 1, :discard)
      |> Nx.tensor
      |> Nx.divide(word_count)
      |> Nx.reshape({:auto, @sequence_length, 1})
      |> Nx.to_batched_list(@batch_size)


    train_labels =
      labels
      |> Enum.chunk_every(@sequence_length, 1, :discard)
      |> Nx.tensor
      |> Nx.reshape({:auto, @sequence_length, 1})
      |> Nx.to_batched_list(@batch_size)


      {train_data, train_labels}
  end


  def transform_text_2(chars, count, char_to_idx) do

    train_data =
      chars
      |> Enum.map(&Map.fetch!(char_to_idx, &1))
      |> Enum.chunk_every(@sequence_length, 1, :discard)
          # We don't want the last chunk since we don't have a prediction for it.
      #|> Enum.drop(-1)
      |> Nx.tensor()
      #|> Nx.divide(count)
      |> Nx.reshape({:auto, @sequence_length, 1})


    train_labels =
      chars
      |> Enum.drop(@sequence_length)
      |> Enum.map(&Map.fetch!(char_to_idx, &1))
      |> Nx.tensor()
      |> Nx.reshape({:auto, 1})
      |> Nx.equal(Nx.iota({count}))


      {train_data, train_labels}
  end

  def predict(init_sequence, params, model) do

        Enum.reduce(1..100, {[], []}, fn _, seq ->
          p = Axon.predict(model, params, init_sequence) |> IO.inspect
          tchar =
            p
            |> Nx.argmax(axis: -1)
            |> Nx.sum()
            |> Nx.to_number()
            |> IO.inspect

          char =
            p
            |> Nx.sum()
            |> Nx.to_number()
            |> IO.inspect

          {elem(seq, 0) ++ [tchar], elem(seq, 1) ++ [char]}
        end)
        |> IO.inspect

  end

  def run do


    normalized_train_data_series = TextEntity.from_dataset("./data/train.txt")
    normalized_test_data_series = TextEntity.from_dataset("./data/test.txt")
    normalized_validate_data_series = TextEntity.from_dataset("./data/valid.txt")

    {char_label_data, unique_words, unique_labels} =
      Streamer.from_file("./data/train.txt")
      |> DataSeries.normalize

    {max_length, max_word} = DataSeries.max_len(unique_words)

    # UNPADDED
    {tokenized_chars, tokenized_labels, {char_to_idx, idx_to_char}, {label_to_idx, idx_to_label}, word_label_array} = Dictionary.from_dataseries(normalized_train_data_series, [verbose: true])


    {tokenized_test_chars, tokenized_test_labels, _char_idx_dict, _charlabel_dict, _word_label_array} = Dictionary.from_dataseries(normalized_test_data_series, [verbose: true])

    {tokenized_validate_chars, _tokenized_labels, _char_idx_dict, _charlabel_dict, _word_label_array} = Dictionary.from_dataseries(normalized_validate_data_series, [verbose: true])

    {input_count, word_char_count, label_count} = DataSeries.stats(tokenized_chars, char_to_idx, label_to_idx)

    IO.puts "Axon has found #{input_count} Words, #{word_char_count} Unique Words &  #{label_count} Unique Labels, Longest Word: #{max_length}::#{max_word}"


    ## With PAdding
    #with_padding =
    #  DataSeries.word_to_chars(word_label_array)
    #  |> padded_batches(max_length, word_char_count, label_count)


      #{padded_train_data, padded_train_labels} = padded_transform(with_padding, #char_to_idx, label_to_idx, max_length)


    #p_tokenized_chars |> Enum.to_list |> IO.inspect

    #  word_label_idx = DataSeries.tokenize(normalized_data_series, char_to_idx, #label_to_idx) |> IO.inspect

    #  count = word_to_idx |> Map.keys |> Enum.map(&String.split(&1, "")) |> #List.flatten |> Enum.uniq |> Enum.count |> IO.inspect

    #  chars = DataSeries.to_chars(normalized_data_series)
    #  char_to_idx = Enum.with_index(chars) |> Enum.into(%{})
      # And a reverse mapping to convert back to characters
      #idx_to_char = Enum.with_index(chars, &{&2, &1}) |> Enum.into(%{})


      {train_data, train_labels} = transform_text({tokenized_chars, tokenized_labels}, {word_char_count, label_count})

      {test_data, test_labels} = transform_text({tokenized_test_chars, tokenized_test_labels}, {word_char_count, label_count})



    #  padded_model = build_model(word_char_count, label_count, max_length+1)
      unpadded_model = build_model(word_char_count, label_count, @sequence_length)
    #  IO.inspect padded_model
    #  IO.inspect unpadded_model

      data = Stream.zip(train_data, train_labels)

      params = train(unpadded_model, data)

      i = Enum.fetch!(test_data, 0)

      predict(i, params, unpadded_model)


  end
end
