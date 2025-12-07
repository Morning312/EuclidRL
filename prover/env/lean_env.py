from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from lean_dojo import Dojo, Environment, ProofResult, TacticState


@dataclass
class GoalSample:
    theorem: str
    premises: list[str]
    prompt: str


class LeanProofEnv:
    """
    Thin wrapper over LeanDojo providing reset/step for RL.
    """

    def __init__(
        self,
        repo: str,
        commit: str,
        theorems: Iterable[str],
        max_steps: int = 64,
        workdir: str | Path = ".lean_env",
    ) -> None:
        self.repo = repo
        self.commit = commit
        self.theorems = list(theorems)
        self.max_steps = max_steps
        self.workdir = Path(workdir)
        self._env: Optional[Environment] = None
        self._dojo: Optional[Dojo] = None
        self._state: Optional[TacticState] = None
        self._steps = 0

    def start(self) -> None:
        env = Environment.from_git_repo(self.repo, self.commit)
        self.workdir.mkdir(parents=True, exist_ok=True)
        dojo = Dojo(env, workdir=str(self.workdir))
        self._env = env
        self._dojo = dojo

    def reset(self, theorem_name: str) -> TacticState:
        if self._dojo is None:
            self.start()
        assert self._dojo is not None
        state, _ = self._dojo.get_state(theorem_name)
        self._state = state
        self._steps = 0
        return state

    def step(self, tactic: str) -> tuple[TacticState, float, bool, ProofResult]:
        """
        Apply a tactic, return next state, reward, done, and lean result.
        """
        assert self._dojo is not None and self._state is not None
        result = self._dojo.run_tac(self._state, tactic)
        self._steps += 1
        done = result.proved or self._steps >= self.max_steps or result.crashed
        reward = 1.0 if result.proved else 0.0
        self._state = result.tactic_state
        return result.tactic_state, reward, done, result
