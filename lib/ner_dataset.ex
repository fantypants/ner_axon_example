defmodule NerDataSet do

  def gen_train_series(filename, opts \\ []) do
    input_columns = Keyword.fetch!(opts, :input_columns)

    {token_array, entity_array} =
      filename
      |> File.stream!()
      |> dict_convert


    Dictionary.from_vocab_labels(token_array, entity_array)

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
      [token | entities] = String.split(line, " ")


      {tokens, ents} = acc

      if(!is_nil(token) && token !== "" && line !== "\n")do
        tokens = tokens ++ [token]
        [ r | rest ] = entities |> List.last |> String.split("\n")

        ents = ents ++ [r]
        acc = {tokens, ents}
      else
        acc
      end




    end) |> IO.inspect
  end



end
