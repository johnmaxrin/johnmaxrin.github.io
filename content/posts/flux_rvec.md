+++
title = "Teaching Rust Not to Panic"
description = "Formal Verification, Liquid Types, Flux"
date = 2026-10-27
draft = false
template = "post.html"
+++

# Teaching Rust Not to Panic: A Refined Vector with Flux

We've all been there. You write `v[i]`, it compiles, you run it, and boom: index out of bounds. Rust's borrow checker is great, but it doesn't know how long your vector is.

What if it did?

That's what [Flux](https://github.com/flux-rs/flux) is for. Flux is a refinement type checker for Rust. You attach little logical facts (refinements) to your types, like "this vector has length `n`", and Flux checks them at compile time. In this post we'll build a **refined vector** (`RVec`) from scratch and use it to verify some classic programs: sum, fibonacci, and binary search.

Flux comes out of research, too. It was introduced in the PLDI 2023 paper [*Flux: Liquid Types for Rust*](https://plsyssec.github.io/cse291y-fall26/papers/flux.pdf) by Nico Lehmann, Adam T. Geller, Niki Vazou, and Ranjit Jhala. We'll point back to it as we go, and there's a short "go read the paper" section at the end.

Let's go.

---

## Setting up the `RVec`

```rust
use flux_rs::*;
```

First, the struct itself. Two attributes do the heavy lifting here.

`#[opaque]` is like telling the verifier that you don't worry about the implementation of the struct. Just use the refinements and the specs provided. The value of the index will be tracked solely by the API for that struct.

`#[refined_by(len: int)]` says our `RVec` (Refined Vector) is refined by a `len` index. `len` helps us reason about the size of the vector. For example, `RVec<i32>[10]` represents a vector of size 10.

```rust
#[opaque]
#[refined_by(len: int)]
pub struct RVec<T> {
    inner: Vec<T>,
}
```

Under the hood it's just a plain old `Vec<T>`. The magic is all in the specs we put on its methods.

> **From the paper:** this is the same refined vector API the authors present in Section 2.3 (Figure 3). One small difference: the paper writes specs with `#[flux::sig(...)]`, while the code here uses the `#[spec(...)]` attribute from `flux_rs`, so don't be thrown by the different syntax.

---

## The API, one spec at a time

We mark the `impl` as `#[trusted]`. That means Flux won't try to verify the method bodies. It takes our word for it and uses the specs as ground truth. (So be honest in your specs!)

### `new`

When creating a new vector, its size is 0, hence the refinement `len` is 0.

```rust
#[trusted]
impl<T> RVec<T>
{
    #[spec(fn() -> RVec<T>[0])]
    pub fn new() -> Self
    {
        Self{inner: Vec::new()}
    }
```

### `push`

Pushing adds one element. We take the vector at some length `n`, and when `push` is done, it's at `n+1`.

(Why can `push` change the length in its type while `len` and `get_mut` can't? The paper calls this a *strong update*, and it's the thing that makes Flux work on imperative Rust. Plain `&mut` references can only do weak updates, meaning the type stays put. A strong reference, written `&strg` in the paper, lets the type change after the call. The paper marks that with `&strg` and an `ensures` clause; in the spec above, it's the `ensures` clause that tells Flux the type changes.)

```rust
    #[spec(fn(self: &mut RVec<T>[@n], T) ensures self:RVec<T>[n+1])]
    pub fn push(&mut self, item: T)
    {
        self.inner.push(item);
    }
```

### `pop`

For pop we have something like `n > 0`, as you cannot pop from a vector of len 0! The precondition goes right into the type of `self`, and afterwards the length drops to `n-1`.

```rust
    #[spec(fn(self: &mut {RVec<T>[@n] | n > 0}) -> T 
        ensures self:RVec<T>[n-1])]
    pub fn pop(&mut self) -> T
    {
        self.inner.pop().unwrap()
    }
```

### `len`

We keep the `len` index as the return value of this function so that Flux can reason about the value returned from this fn. In other words: the `usize` you get back is *exactly* `n`.

```rust
    #[spec(fn(self: &RVec<T>[@n]) -> usize[n])]
    pub fn len(&self) -> usize
    {
        self.inner.len()
    }
```

### `get` and `get_mut`

The index passed to the `get` fn should be less than the `len` index of the `RVec`. This is the one that kills out-of-bounds errors for good.

```rust
    #[spec(fn(self: &RVec<T>[@n], i:usize{i < n}) -> &T)]
    pub fn get(&self, i:usize) -> &T
    {
        &self.inner[i]
    }

    #[spec(fn(self: &mut RVec<T>[@n], i:usize{i < n}) -> &mut T)]
    pub fn get_mut(&mut self, i:usize) -> &mut T
    {
        &mut self.inner[i]
    }
    
}
```

### Making it feel like Rust: `Index` and `IndexMut`

Calling `.get(i)` everywhere is a bit clunky. Let's implement `Index` and `IndexMut` so we can write `vec[i]` and still get the bounds check at verification time.

```rust
#[trusted]
impl<T> std::ops::Index<usize> for RVec<T>
{
    type Output = T;

    #[trusted_impl]
    #[spec(fn(self: &RVec<T>[@n], index:usize{index < n}) -> &T)]
    fn index(&self, index: usize) -> &T
    {
        &self.inner[index]
    }
} 

#[trusted]
impl<T> std::ops::IndexMut<usize> for RVec<T>
{
    #[trusted_impl]
    #[spec(fn(self: &mut RVec<T>[@n], index:usize{index < n}) -> &mut T)]
    fn index_mut(&mut self, index: usize) -> &mut T
    {
        &mut self.inner[index]
    }
}
```

That's the whole library. Now let's actually use it.

---

## Test drive: push and pop

Here we promise that `test_push` returns an empty vector, `RVec<i32>[0]`. Flux follows the length through every call.

```rust
#[spec(fn() -> RVec<i32>[0])]
fn test_push() -> RVec<i32>
{
    // create an RVec with type RVec<i32>[0]
    let mut vec = RVec::new(); 

    vec.push(1); // RVec<i32>[1]
    vec.push(1); // RVec<i32>[2]
    vec.push(1); // RVec<i32>[3]
    assert(vec.len() == 3);
    vec.pop(); // RVec<i32>[2]
    vec.pop(); // RVec<i32>[1]
    vec.pop(); // RVec<i32>[0]
    assert(vec.len() == 0);

    vec
}
```

Three pushes, three pops, and Flux agrees we end up back at zero. Try adding a fourth `pop()` and watch it complain.

---

## Loops: `zeros`

Loops are where verification usually gets spicy, but Flux handles this nicely. We promise that `zeros(n)` returns a vector of length exactly `n`.

```rust
#[spec(fn(n:usize) -> RVec<i32>[n])]
fn zeros(n:usize) -> RVec<i32>
{
    let mut i = 0;
    let mut v = RVec::new();
    while i<n
    {
        v.push(0);
        i += 1;
    }

    v
}
```

Flux infers the loop invariant on its own: at every iteration, `v` has length `i`, and `i <= n`. When the loop exits, `i == n`, so `v` has length `n`. No annotations needed.

> **From the paper:** this is the `init_zeros` example from Section 2.3. The authors call the technique *liquid inference*: Flux invents unknown predicates for the loop, turns the program into a set of constraints, and hands them to a solver (Section 4.2). That's why you never write a loop invariant by hand. In their benchmarks, the comparable verifier Prusti needed hand-written loop invariants taking up as much as 24% of the code (14% on average), and Flux needed none.

---

## Summing a vector

Here's the classic. Walk the vector and add things up.

Try the below code with `i <= vec.len()`. You'll get an error saying it can't verify the post condition. (Of course! At `i == vec.len()` you'd be reading one past the end.)

```rust
fn sum_vec(vec: &RVec<i32>) -> i32 {
    let mut res = 0;
    let mut i = 0;
    while i < vec.len() {
        res += vec.get(i);
        i += 1;
    }
    res
}
```

And the same thing, but nicer, using `[..]`:

```rust
// Uses [..]
fn sum_vec_nice(vec: &RVec<i32>) -> i32 {
    let mut res = 0;
    let mut i = 0;
    while i < vec.len() {
        res += vec[i];
        i += 1;
    }
    res
}
```

Same guarantee, much friendlier syntax.

---

## Fibonacci

Let's build fib with a vector as a table. Either add the spec, or in the code add a check: if `n <= 0`, don't pop. Try removing the spec and go through the error message. It's a good one! Flux tells you the `pop` might happen on an empty vector.

With the spec, we push the burden onto the caller: `n` must be greater than 0.

```rust
#[spec(fn(n:usize{n > 0}) -> i32)]
pub fn fib(n: usize) -> i32
{
    let mut r = RVec::new();
    let mut i = 0;

    
    while i < n 
    {
        if i == 0
        { r.push(0); }

        else if i == 1
        { r.push(1); }

        else 
        {
            let a = r[i-1];
            let b = r[i-2];

            r.push(a+b);
        }
        i += 1;
    }

    r.pop()
}
```

Notice how `r[i-1]` and `r[i-2]` are verified too. By the time we're at `i >= 2`, the vector has length `i`, so both indices are in bounds. And since `n > 0`, the loop runs at least once, so the final `pop()` is safe.

---

## Binary search

Now for the boss fight.

If we remove check 1 or check 2, you'll get an error from Flux saying it cannot verify conditions. Try it and see!

```rust
pub fn binary_search(vec: &RVec<i32>, x: i32) -> Result<usize, usize> {
    let mut size = vec.len();
    let mut left = 0;
    let mut right = size;
    let mut val = 0;
    while left <= right {
        let mid = left + size / 2;

        // Check 1
        if mid < vec.len()
        { val = vec[mid]; }

        if val < x {
            left = mid + 1;
        } else if x < val {
            right = mid;
        } else {
            return Ok(mid);
        }

        // Check 2
        if left <= right
        { size = right - left; }
        else 
        { size = 0 }
    }
    Err(left)
}
```

So what are those checks doing?

- **Check 1** guards the read `vec[mid]`. Flux can't prove `mid < vec.len()` on its own, because the loop condition is `left <= right`, which technically allows `mid` to land on the end. So we check it ourselves and Flux is happy.
- **Check 2** guards the subtraction `right - left`. Since these are `usize`s, subtracting a bigger number from a smaller one underflows. The check makes sure that never happens.

Remove either one and Flux will point right at the spot where things could go wrong. That's kind of the whole point.

> **From the paper:** binary search (`bsearch`) is one of the benchmarks in the evaluation (Section 5), ported from an earlier refinement type system for OCaml. The goal there is exactly what we did here: prove that every vector access is in bounds.

---

## Wrapping up

Here's what we did:

1. Built `RVec`, a vector that carries its length in its type.
2. Wrote specs for `new`, `push`, `pop`, `len` and `get`, so Flux can track length through the whole program.
3. Verified loops, indexing, fibonacci, and binary search, with no out-of-bounds, no popping from empty, and no underflow.

The trick is the split: a small, **trusted** core (`RVec`) that you audit carefully, and everything built on top gets checked automatically. Break a rule, and Flux tells you exactly where.

Go play with it. Change `<` to `<=`, remove a spec, delete a check, and read the error messages. That's the fastest way to get a feel for refinement types.

---

## Go read the paper

If this made you curious about *why* it works, the paper is worth your time:

**[Flux: Liquid Types for Rust](https://plsyssec.github.io/cse291y-fall26/papers/flux.pdf)**
Nico Lehmann, Adam T. Geller, Niki Vazou, and Ranjit Jhala. *Proc. ACM Program. Lang. 7, PLDI, Article 169* (June 2023). DOI: [10.1145/3591283](https://doi.org/10.1145/3591283)

Here's a map of what's inside:

- **Section 2: A Tour of Flux.** Indexed types, existential types, and how ownership and borrowing fit with refinements. This is where the `RVec` API comes from, and it's the friendliest place to start.
- **Section 3: Formalization.** The core calculus (λLR) and a soundness proof, which says that well-typed programs don't get stuck, so vectors are always accessed with valid indices.
- **Section 4: Algorithmic Verification.** How the Rust compiler plugin turns all this into constraints and infers refinements, so you don't have to write loop invariants.
- **Section 5: Evaluation.** Flux versus Prusti, a program-logic verifier for Rust. Flux cut specification size by about half and verification time by an order of magnitude. For example, `fft` took Flux about 0.7 seconds and Prusti about 167. The paper is also upfront about the trade-off: Prusti can express deeper properties, like sortedness, that Flux can't.

The source code is on GitHub at [flux-rs/flux](https://github.com/flux-rs/flux).

Happy verifying!