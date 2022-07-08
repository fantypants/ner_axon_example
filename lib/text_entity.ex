defmodule TextEntity do
  @column_names %{
    "Index" => 0,
    "Word" => 1,
    "Phrase" => 2
  }

  @def_split " "

  def to_dictionary(data) do
    # Create dict

  end




  def from_dataset(filename, opts \\ []) do

    data_stream =
      filename
      |> File.stream!()

      data_stream
      |> Stream.map(fn line ->
        [token | entities] = String.split(line, @def_split)

        if(!is_nil(token) && token !== "" && line !== "\n")do

          [ r | rest ] = entities |> List.last |> String.split("\n")
        #  |> Enum.map(fn str ->
        #    [s | rest] =
        #    s
        #  end)
          token_size = String.length(token)

          ent_range = Range.new(1, token_size)

          entity_padded = ent_range |> Enum.map(fn index -> r end)


          [token, entity_padded ++ ["O"]]
        else
          nil
        end

      end)
      |> Enum.take(256) # Slice a small sample
      |> Stream.reject(fn x -> is_nil(x) end)
  end


  #splitter => " " || "\t"
  defp split_line(line, acc, splitter) do

    [token | entities] = String.split(line, splitter)

    {tokens, ents} = acc

    if(!is_nil(token) && token !== "" && line !== "\n")do

      tokens = tokens ++ [token]


      [ r | rest ] = entities |> List.last |> String.split("\n")
    #  |> Enum.map(fn str ->
    #    [s | rest] =
    #    s
    #  end)
      token_size = String.length(token)

      ent_range = Range.new(1, token_size)

      entity_padded = ent_range |> Enum.map(fn index -> r end)

      ents = ents ++ entity_padded ++ ["O"]
      acc = {tokens, ents}
    else
      acc
    end

  end

  def gen_train_series(filename, opts \\ []) do
    input_columns = Keyword.fetch!(opts, :input_columns)

    {token_dict, entity_dict} =
      filename
      |> File.stream!()
      |> Enum.take(1024)
      |> dict_convert
      |> Dictionary.create



  #  {test_inputs, train_inputs} =
    #  token_dict.character_elements
    #  |> Enum.map(fn i ->
    #    Nx.tensor(i)
    #  end)



  end



  # convert to the tuple array
  def dict_convert(lines) do
    lines
    |> Enum.reduce({[], []}, fn line, acc ->
      case line do
        nil -> acc
        _-> split_line(line, acc, " ")
      end
    end)
  end

  def split_inputs(filename, opts, :json) do
    input_columns = Keyword.fetch!(opts, :input_columns)
    target_column = Keyword.fetch!(opts, :target_column)
    test_train_ratio = Keyword.fetch!(opts, :test_train_ratio)

    parsed =
      filename
      |> File.read!()
      |> Poison.Parser.parse!
      |> Enum.map(&convert_raw/1)
      |> Enum.map(&convert/1)
      |> Enum.reject(&is_nil/1)
      |> IO.inspect

    {test_inputs, train_inputs} =
      parsed
      #|> slice_columns(input_columns)
    #  |> HackyTools.hacky_normalize_words()
      |> Enum.map(fn i ->
        Nx.tensor([i])

      end)
      |> split(test_train_ratio)

    {test_targets, train_targets} =
      parsed
      |> slice_columns([target_column])
      |> Enum.map(fn a ->
        Nx.tensor([a])
      end)
      |> split(test_train_ratio)

    {
      test_inputs,
      test_targets,
      train_inputs,
      train_targets
    }
  end

  defp split(rows, ratio) do
    count = Enum.count(rows)
    Enum.split(rows, ceil(count * ratio))
  end

  def convert_raw(line) do

  %{"url" => url, "text_results" => results} = line

   results =
     results
     |> Enum.reduce([], fn %{"textContent" => phrase, "textSegmentAnnotations" => annotations}, acc ->


       all =
       annotations
       |> Enum.reduce([], fn %{"startOffset" => wrd_start, "endOffset" => wrd_stp}, annotations ->


         word =
           Range.new(wrd_start, wrd_stp)
           |> Enum.reduce([], fn idx, wrd ->
             IO.inspect idx
             wrd ++ [String.at(phrase, idx)]
           end)
           |> IO.inspect
           |> Enum.join

          annotations ++ [0, word, phrase]

       end)


       all ++ acc

   end)


  #  [
  #    index,
  #    word |> String.trim() |> String.length,
  #    phrase |> String.trim() |> String.length
  #  ]

  rescue
    e in ArgumentError ->
      unless e.message =~ "not a textual representation of a float" do
        raise e
      end
  end

  def convert(line) do

  %{"Phrase" => phrase, "Word" => word, "Index" => index} = line


    [
      index,
      word |> String.trim() |> String.length,
      phrase |> String.trim() |> String.length
    ]

  rescue
    e in ArgumentError ->
      unless e.message =~ "not a textual representation of a float" do
        raise e
      end
  end

  defp slice_columns(data, columns) do

    column_indexes =
      columns
      |> Enum.map(fn name -> Map.get(@column_names, name) end)
      |> Enum.reverse()

    data
    |> Stream.map(fn row ->
      column_indexes
      |> Enum.reduce([], fn index, out ->
        IO.inspect index
        IO.inspect out
        value = Enum.at(row, index)
        [value | out]
      end)
    end)
  end





  def split_char_labels(tokens, labels) do
   '''
   For a given input/output example, break tokens into characters while keeping
   the same label.
   '''

   input_chars = []
   output_char_labels = []

   tokens_labels = Enum.zip(tokens, labels)


   Enum.reduce(tokens_labels, {tokens, labels}, fn {tkn, lbl} ->
     IO.inspect tkn
     IO.inspect lbl
     #List.append(input_chars

   end)

   []

   #for token,label in zip(tokens,labels):

  #   input_chars.extend([char for char in token])
  #   input_chars.extend(' ')
  #   output_char_labels.extend([label]*len(token))
  #   output_char_labels.extend('O')

  # return [[char2idx[x] for x in input_chars[:-1]],np.array([label2idx[x] for x #in output_char_labels[:-1]])]



end

end
