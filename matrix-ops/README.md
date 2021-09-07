# numc

### Provide answers to the following questions for each set of partners.
- How many hours did you spend on the following tasks?
  - Task 1 (Matrix functions in C): 25 hours
  - Task 2 (Writing the Python-C interface): 8 hours
  - Task 3 (Speeding up matrix operations): 22 hours
- Was this project interesting? What was the most interesting aspect about it?
  - <b>The most interesting part to me about this project was that it involved getting accustomed to the inner workings of an API. Never before had I worked with an external codebase that is independent from the skeleton code for school projects.  While a bit complicated to navigate at first, I found that working with the C-Python interface was a good lesson in how the strengths of different programming languages can be combined to create one cohesive program that benefits from those strengths. The numc module I wrote in C worked exactly as expected in the Python interpreter, throwing the appropriate error messages when necessary. I found it pretty surprising how the API seamlessly overrides the default methods associated with certain operators with the methods I wrote in C.  </b>
- What did you learn?
  - <b>A big conclusion I derived from working on this project was that arriving at a solution to a particular problem does not mean that you reached the most efficient one. This was very apparent when I modified my implementation for the pow_matrix method. Upon changing my naive approach to the approach of exponentiation by repeated squaring, my solution automatically became 180 times faster without needing to change my naive implementation for matrix multiplication. In addition to that, working with Intel intrinsics (particularly the AVX set) was a great lesson in the power of parallelism. In some cases, applying these to an algorithm is a fairly straightforward process, like for example, the add_matrix and abs_matrix methods. In other cases, such as mul_matrix, where the algorithm is bit more complicated, incorporating intrinsics was definitely a tough mental exercise. I would go as far as to say that the hardest part of this project was figuring out how to optimize mul_matrix using these intrinsics. It took me quite a while to figure out when and where it would be best to load input matrix values into vectors for optimization operations. However, getting through all that trouble resulted in sign in significant speedup for both pow_matrix and mul_matrix (413x for pow, 31x for mul). </b>
- Is there anything you would change?
  - <b>I would have loved to have gone further with loop unrolling in all of my methods and see that would further increase speedup in conjuction with the intrinsics I included. However, due to time constraints, I was not able to fully explore every option that could have potentially lead to greater speedups.  </b>

### If you worked with a partner:
- In one short paragraph, describe your contribution(s) to the project.
  - <b>N/a</b>
- In one short paragraph, describe your partner's contribution(s) to the project.
  - <b>N/a</b>