defmodule Streamer do

  # File stuff
  def from_file(filename, opts \\ []) do

    data_stream =
      filename
      |> File.stream!()
  end

end
