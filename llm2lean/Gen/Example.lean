import Mathlib

theorem Theorem_AddComm (a b : ℝ) : a + b = b + a := by
  simpa [add_comm]
