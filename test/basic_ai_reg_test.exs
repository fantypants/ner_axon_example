defmodule BasicAiRegTest do
  use ExUnit.Case
  doctest BasicAiReg

  test "greets the world" do
    assert BasicAiReg.hello() == :world
  end
end
