defmodule BasicAiReg.MixProject do
  use Mix.Project

  def project do
    [
      app: :basic_ai_reg,
      version: "0.1.0",
      elixir: "~> 1.11",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [

    {:exla, path: "../nx/exla"},
    {:nx, "~> 0.2", override: true},
    {:flow, "~> 1.1.0"},
    {:pixels, "~> 0.2.0"},
    #{:req, "~> 0.3.0"},
    {:kino, "~> 0.3.1"},
    {:explorer, "~> 0.2.0"},
    {:telemetry_metrics, "~> 0.6"},
    {:telemetry_poller, "~> 1.0"},
    {:scidata, "~> 0.1.6"},
    {:poison, "~> 2.2.0", override: true},
    {:axon, "~> 0.1.0"},
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
    ]
  end
end
