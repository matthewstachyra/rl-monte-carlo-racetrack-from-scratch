# From scratch - On Policy Monte Carlo Control

Below is the racetrack the agent learns an optimal policy to get from the start line to the finish line. The `s` character represents any via start position; the `f` character represents any viable finish position; `1` represents a valid part of the track that the agent can move to while `0` represents an invalid part of the track, or what is out of bounds.
	
![Alt text](/screenshots/optimalrun1.gif)

This repo uses pipenv but all that's required is `numpy` to get this running.

`python main` or `pipenv shell` followed by pipenv run python main`
