defmodule Dictionary do

  def from_dataseries(token_label_array, opts \\ []) do
    IO.inspect opts

    full_array =
      token_label_array
      |> Stream.map(fn [token, labels] ->


          case token do
          [t | rest] ->


          #  text_array =
              #split_text(token)
              #|> Enum.reject(fn str -> str == "" end)

              [token, labels]

          _->
          text_array =
            split_text(token)
            |> Enum.reject(fn str -> str == "" end)

            [text_array ++ [""], labels]

          end



      end)
      |> Stream.reject(fn x -> is_nil(x) end)


      token_array =
      full_array
        |> Stream.flat_map(fn [t, l] -> t end)

      label_array =
      full_array
        |> Stream.flat_map(fn [t, l] -> l end)

      token_idx =
        token_array
        |> Stream.uniq
        |> Stream.with_index
        |> Enum.into(%{})

      label_idx =
        label_array
        |> Stream.uniq
        |> Stream.with_index
        |> Enum.into(%{})


      idx_to_token =
          token_array
          |> Stream.uniq
          |> Enum.with_index(&{&2, &1})
          |> Enum.into(%{})

      idx_to_label =
        label_array
        |> Stream.uniq
        |> Enum.with_index(&{&2, &1})
        |> Enum.into(%{})


        token_char_idx =
          token_array
          |> Stream.map(fn tk ->
            if(opts[:token_dict]) do
                Map.get(opts[:token_dict], tk, 0) # supplied
            else
                Map.get(token_idx, tk, 0) # Generated dict
            end
          end)
          |> Enum.to_list
          |> IO.inspect

        label_char_idx =
            label_array
            |> Stream.map(fn tk ->
              if(opts[:label_dict]) do
                  Map.get(opts[:label_dict], tk, 0) # supplied
              else
                  Map.get(label_idx, tk, 0) # Generated dict
              end
            end)






            if(opts[:verbose])do
             {
               token_char_idx,
               label_char_idx,
               {token_idx, idx_to_token},
               {label_idx, idx_to_label},
               full_array
             }
           else
             {
               token_char_idx,
               label_char_idx,
               {token_idx, idx_to_token},
               {label_idx, idx_to_label},
             }
           end




  end

  def from_token_label_stream(token_label_array) do

    full_array =
      token_label_array
      |> Stream.map(fn [token, labels] ->

        text_array =
          split_text(token)
          |> Enum.reject(fn str -> str == "" end)

          [text_array ++ [""], labels]


      end)


      token_array =
      full_array
        |> Stream.flat_map(fn [t, l] -> t end)

      label_array =
      full_array
        |> Stream.flat_map(fn [t, l] -> l end)

      token_idx =
        token_array
        |> Stream.uniq
        |> Stream.with_index
        |> Enum.into(%{})






      label_idx =
        label_array
        |> Stream.uniq
        |> Stream.with_index
        |> Enum.into(%{})




        token_char_idx =
          token_array
          |> Stream.map(fn tk ->
            Map.get(token_idx, tk)
          end)

        label_char_idx =
            label_array
            |> Stream.map(fn tk ->
              Map.get(label_idx, tk)
            end)




             %{
               token_char_idx: token_char_idx,
               label_char_idx: label_char_idx,
               token_idx: token_idx,
               label_idx: label_idx
             }




  end


  def from_vocab_labels(token_array, entity_array) do

    tokens =
      token_array
      |> Enum.uniq
      |> Enum.map(fn str -> String.lowercase(str) end)

    labels =
      entity_array
      |> Enum.uniq

  end

  def create({token_array, entity_array}) do
    # tokenize arrays
    tokenized_tokens = tokenize(token_array)


    # Turn chars to the id dict
    token_dict = set_element_ids(tokenized_tokens)
    entity_dict = set_element_ids(entity_array)

    {token_dict, entity_dict}
  end

  defp tokenize(token_array) do
    # [a,b,c,d]
    token_array
    |> Enum.reduce([], fn token, character_list ->

      text_array =
        split_text(token)
        |> Enum.reject(fn str -> str == "" end)

        character_list ++ text_array ++ [""]
    end)


  end

  defp split_text(text), do: text |> String.split("") #|> any_list
  defp set_element_ids(array) do
    array
    |> Enum.reduce(%{dictionary: %{}, character_elements: []}, fn character_element, acc ->
      %{dictionary: dict, character_elements: chars} = acc

      length = Map.keys(dict) |> Enum.count
      exists? = Map.has_key?(dict, character_element)


      if exists? do
        char_codepoint = Map.get(dict, character_element)
        new_char_list = chars ++ [char_codepoint]
        %{dictionary: dict, character_elements: new_char_list}


      else
        new_dict = Map.put(dict, character_element, length + 1)
        char_codepoint = Map.get(new_dict, character_element)
        new_char_list = chars ++ [char_codepoint]

        %{dictionary: new_dict, character_elements: new_char_list}

      end

    end)

  end



  defp set_element_ids(array) do
    array
    |> Enum.reduce(%{dictionary: %{}, character_elements: []}, fn character_element, acc ->
      %{dictionary: dict, character_elements: chars} = acc

      length = Map.keys(dict) |> Enum.count
      exists? = Map.has_key?(dict, character_element)


      if exists? do
        char_codepoint = Map.get(dict, character_element)
        new_char_list = chars ++ [char_codepoint]
        %{dictionary: dict, character_elements: new_char_list}


      else
        new_dict = Map.put(dict, character_element, length + 1)
        char_codepoint = Map.get(new_dict, character_element)
        new_char_list = chars ++ [char_codepoint]

        %{dictionary: new_dict, character_elements: new_char_list}

      end

    end)

  end
end
