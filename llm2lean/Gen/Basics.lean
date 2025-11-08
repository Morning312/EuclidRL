import Mathlib

-- a simple function on naturals with pattern matching
def double : Nat → Nat
| 0       => 0
| (n+1)   => double n + 2

#eval double 5   -- 10

-- a tiny structure and a function on it
structure Pt where
  x : ℝ
  y : ℝ
deriving Repr

def dist2 (p q : Pt) : ℝ := (p.x - q.x)^2 + (p.y - q.y)^2

#eval dist2 ⟨0,0⟩ ⟨3,4⟩    -- 25
