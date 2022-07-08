defmodule BiNerWord do
  require Axon
  # From Example http://alexminnaar.com/2019/08/22/ner-rnns-tensorflow.html
  EXLA.set_as_nx_default([:tpu, :cuda, :rocl, :host]) # My :host option doesnt work, jit_apply is undefined
  require Logger
  @sequence_length 100
  @batch_size 128
  @lstm 1024
  @embed_dimension 256

  # seqlen
  # word_count


  def train(model, data) do

    model
    |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adam(0.001))
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(data, %{}, epochs: 1, iterations: 1)

  end

  def build_model(word_count, label_count) do

      Axon.input({nil, @sequence_length, 1}, "inputs")
       |> Axon.embedding(word_count, @sequence_length)
       |> Axon.nx(fn t ->
         t[[0..-1//1, -1]]
     end)
       |> Axon.spatial_dropout(rate: 0.1)
       |> Axon.lstm(@sequence_length)
       |> then(fn {{new_cell, new_hidden}, out} ->
         out
       end)
      |> Axon.dropout(rate: 0.2)
      |> Axon.dense(label_count, activation: :softmax)

  end

  def transform_words(word_labels, word_to_idx, label_to_idx, wcount, lcount) do

    train_data =
      word_labels
      |> Enum.map(fn {word, label} ->

        with {:ok, id} <- Map.fetch(word_to_idx, word) do
          id
        else
          _->
          0
        end
      end)
      |> Enum.chunk_every(@sequence_length, 1, :discard)
      |> Nx.tensor
      #|> Nx.divide(wcount)
      |> Nx.reshape({:auto, @sequence_length, 1})
      |> Nx.to_batched_list(@batch_size)


    train_labels =
      word_labels
      |> Enum.map(fn {word, label} ->


        with {:ok, id} <- Map.fetch(label_to_idx, label) do
          id
        else
          _->
          0
        end
      end)
      |> Enum.chunk_every(@sequence_length, 1, :discard)
      |> Nx.tensor
    #  |> Nx.divide(lcount)
      |> Nx.reshape({:auto, @sequence_length, 1})
      |> Nx.to_batched_list(@batch_size)


      {train_data, train_labels}
  end


  def predict(init_sequence, params, model, idx_to_char, idx_to_label, count, real) do

        Enum.reduce(1..100, [], fn _, seq ->

          p = Axon.predict(model, params, init_sequence)



            tchar2 =
              p
              |> Nx.argmax(axis: -1)
              |> Nx.to_flat_list



          fed =
          init_sequence
        #  |> Nx.multiply(count)
          |> Nx.to_flat_list



          rl =
            real
            |> Nx.to_flat_list

          res =
          tchar2

          seq ++ [{fed, res, rl}]

        end)
        |> Enum.map(fn {inp, res, rll} ->
#
          input =
          inp
          |> Enum.map(fn inpseq ->
            Map.fetch!(idx_to_char, inpseq)
          end)




          output =
          res
          |> Enum.map(fn inpseq ->
            Map.fetch!(idx_to_label, inpseq)
          end)






            rl_d =
            rll
            |> Enum.map(fn inpseq ->
              Map.fetch!(idx_to_label, inpseq)
            end)




        {input, output, rl_d}

        end)
        |> Enum.map(fn {i, o, r} ->

              Enum.zip([i, o, r])
              |> Enum.map(fn {input, output, r} ->
                true? = if output == r do "CORRECT" else "INCORRECT" end

                if true? == "CORRECT" do
                  Logger.warn "Word was: #{input}, Predicted Category was: #{output} Real Result: #{r}, RETURNED: #{true?}"
                else
                  Logger.error "Word was: #{input}, Predicted Category was: #{output} Real Result: #{r}, RETURNED: #{true?}"
                end



            end)

        end)


  end

  def load_and_predict do
    {word_label_data, unique_words, unique_labels} =
      Streamer.from_file("./data/train.txt")
      |> DataSeries.normalize

      {test_word_label_data, unique_test_words, unique_test_labels} =
        Streamer.from_file("./data/test.txt")
        |> DataSeries.normalize

      streamed =
      word_label_data
      |> Stream.reject(&is_nil(&1))

      test_streamed =
      test_word_label_data
      |> Stream.reject(&is_nil(&1))


      word_count  = unique_words |> Enum.count
      label_count  = unique_labels |> Enum.count

      word_to_idx =
        unique_words
        |> Stream.uniq
        |> Stream.with_index
      #  |> Enum.map(&IO.inspect(&1))
        |> Enum.into(%{})


        label_to_idx =
          unique_labels
          |> Stream.uniq
          |> Stream.with_index
          |> Enum.into(%{})

          idx_to_word =
              unique_words
              |> Stream.uniq
              |> Enum.with_index(&{&2, &1})
              |> Enum.into(%{})

          idx_to_label =
            unique_labels
            |> Stream.uniq
            |> Enum.with_index(&{&2, &1})
            |> Enum.into(%{})

            {td, tl} = transform_words(streamed, word_to_idx, label_to_idx, word_count, label_count)

            {ttd, ttl} = transform_words(test_streamed, word_to_idx, label_to_idx, word_count, label_count)

            serialized =
              File.read!("./sample")


            Axon.deserialize(serialized) |> IO.inspect

          #  i = Enum.fetch!(td, 3)
          #  real_results = Enum.fetch!(tl, 3)



              #  predict(i, saved_params, saved_model, idx_to_word, idx_to_label, word_count, real_results)



  end

  def run do


  #  normalized_train_data_series = TextEntity.from_dataset("./data/train.txt")


    {word_label_data, unique_words, unique_labels} =
      Streamer.from_file("./data/train.txt")
      |> DataSeries.normalize

      {test_word_label_data, unique_test_words, unique_test_labels} =
        Streamer.from_file("./data/test.txt")
        |> DataSeries.normalize

      streamed =
      word_label_data
      |> Stream.reject(&is_nil(&1))

      test_streamed =
      test_word_label_data
      |> Stream.reject(&is_nil(&1))


      word_count  = unique_words |> Enum.count
      label_count  = unique_labels |> Enum.count

      word_to_idx =
        unique_words
        |> Stream.uniq
        |> Stream.with_index
      #  |> Enum.map(&IO.inspect(&1))
        |> Enum.into(%{})


        label_to_idx =
          unique_labels
          |> Stream.uniq
          |> Stream.with_index
          |> Enum.into(%{})

          idx_to_word =
              unique_words
              |> Stream.uniq
              |> Enum.with_index(&{&2, &1})
              |> Enum.into(%{})

          idx_to_label =
            unique_labels
            |> Stream.uniq
            |> Enum.with_index(&{&2, &1})
            |> Enum.into(%{})

            {td, tl} = transform_words(streamed, word_to_idx, label_to_idx, word_count, label_count)

            {ttd, ttl} = transform_words(test_streamed, word_to_idx, label_to_idx, word_count, label_count)

            unpadded_model = build_model(word_count, label_count)
            IO.inspect unpadded_model
            data = Stream.zip(td, tl)
#
            unpadded_params = train(unpadded_model, data)

            serialized_model = Axon.serialize(unpadded_model, unpadded_params)

            File.write!("./sample", serialized_model, [mode: :raw])
            m = File.read!("./sample")


            Axon.deserialize(m) |> IO.inspect
            #File.write!("./model_b", serialized_model)

            #i = Enum.fetch!(td, 3)
            #real_results = Enum.fetch!(tl, 3)



                #predict(i, unpadded_params, unpadded_model, idx_to_word, #idx_to_label, word_count, real_results)



  end


end
