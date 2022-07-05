defmodule DataSeries do
    @def_split " "
    @start "-DOCSTART- -X- -X- O\n"
    @ending "-------------------------------------------------------------"

  # Tokenization & Padding Module

  # Turn Words into [{words, label}, ..], where: words = String & labels = String
  def normalize(streamed_data, :chars) do

    char_label_data =
    streamed_data
    |> Stream.map(fn line ->
      [token | entities] = String.split(line, @def_split)
      if(!is_nil(token) && token !== "" && line !== "\n")do

        [ r | rest ] = entities |> List.last |> String.split("\n")

        token_size = String.length(token)

        ent_range = Range.new(1, token_size)

        entity_padded = ent_range |> Enum.map(fn index -> r end)

        {token, entity_padded ++ ["O"]}
      else
        nil
      end
    end)
    #|> Enum.take(256) # Slice a small sample
    |> Stream.reject(fn x -> is_nil(x) end)


    unique_words = char_label_data |> Enum.map(&(elem(&1, 0))) |> Enum.uniq


    unique_labels = char_label_data |> Enum.map(&(elem(&1, 1))) |> Enum.uniq

    {char_label_data, unique_words, unique_labels}
  end

  def normalize(streamed_data) do

    char_label_data =
    streamed_data
    |> Stream.map(fn line ->
      [token | entities] = String.split(line, @def_split)
      if(
      !is_nil(token) &&
      token !== "" &&
      line !== "\n" &&
      token !== @start &&
      token !== @ending)
      do

        [ r | rest ] = entities |> List.last |> String.split("\n")

        {token, r}
      else
        nil
      end
    end)
    #|> Enum.take(256) # Slice a small sample
    |> Stream.reject(fn x -> is_nil(x) end)


    unique_words = char_label_data |> Enum.map(&(elem(&1, 0))) |> Enum.uniq


    unique_labels = char_label_data |> Enum.map(&(elem(&1, 1))) |> Enum.uniq

    {char_label_data, unique_words, unique_labels}
  end

  def max_len(words_array) do
    words_array
    |> Enum.reduce({0, ""}, fn wrd, max_len ->
      len = String.length(wrd)
      if len > elem(max_len, 0) do
        {len, wrd}
      else
        max_len
      end
    end)
  end

  def to_dict({streamed_data_array, words, labels}) do

    word_to_idx =
    words
    |> Enum.with_index
    |> Enum.into(%{})


    label_to_idx =
    labels
    |> Enum.with_index
    |> Enum.into(%{})


    idx_to_word =
      words
      |> Enum.with_index(&{&2, &1})
      |> Enum.into(%{})

    idx_to_label =
      labels
      |> Enum.with_index(&{&2, &1})
      |> Enum.into(%{})


      {word_to_idx, label_to_idx, idx_to_word, idx_to_label}

  end

  def tokenize({streamed_array, _, _} = normalized_data_series, word_to_idx, label_to_idx) do

    word_idx =
    streamed_array
    |> Enum.map(&Map.fetch!(word_to_idx, elem(&1, 0)))

    label_idx =
    streamed_array
    |> Enum.map(&Map.fetch!(label_to_idx, elem(&1, 1)))

    {word_idx, label_idx}
  end

  def stats2({token_label_array, char_or_words, labels}) do

    {token_label_array |> Enum.count,
    char_or_words |> Enum.count,
    labels |> Enum.count}
  end


  def stats(token_label_array, vocab_idx, label_idx) do
    label_count = (label_idx |> Map.keys |> Enum.count)
    char_or_words = (vocab_idx |> Map.keys |> Enum.count)

    {token_label_array |> Enum.count,
    char_or_words,
    label_count}
  end

  def word_to_chars(streamed_array) do
    char_label_array =
    streamed_array
    |> Stream.map(fn [token, labels] ->
      text_array =
        token
        #|> String.split("")
        |> Enum.reject(fn str -> str == "" end)

        [text_array ++ [""], labels]
    end)
  #  |> Enum.map(fn x -> IO.inspect x end)





  end

  def to_chars({streamed_array, _words, _labels}) do
    char_label_array =
    streamed_array
    |> Stream.map(fn {token, labels} ->

      text_array =
        token
        |> String.split("")
        |> Enum.reject(fn str -> str == "" end)



        {text_array ++ [""], labels}
    end)
    |> Enum.to_list
    |> IO.inspect


    uniq_chars = char_label_array |> Stream.flat_map(fn {t, l} -> t end) |> Stream.uniq |> IO.inspect


    uniq_labels = char_label_array |> Stream.flat_map(fn {t, l} -> l end) |> Stream.uniq

    {char_label_array, uniq_chars, uniq_labels}
  end

end
