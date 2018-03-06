# Dataset Creation

This directory is used to generate relational questions from [Visual Genome](http://visualgenome.org/).

Each question is input to the network in the form of an array with shape: (2\*num_objects + num_questions).  
```[ Obj_1,..., Obj_n, Sub_1,..., Sub_n, Q_1,..., Q_n ]```

The answer is in the form of an array with shape: (num_objects + 2).  
```[ Obj_1,...,Obj_n, yes, no ]```

O<sub>i</sub> and O<sub>j</sub> are specific objects in the dataset, where i ≠ j. Questions 1-2 are *non-relational*, questions 3-8 are *relational*.

1. Where is the O<sub>i</sub>?
  * **Subject:**  O<sub>i</sub>
  * **Answer:** O<sub>i</sub>
  * **Location:** L<sub>i</sub>
2. Is the O<sub>i</sub> on the left?
  * **Subject:**  O<sub>i</sub>
  * **Answer:** (yes/no)
  * **Location:** L<sub>i</sub>
3. Is  O<sub>i</sub> on the left of O<sub>j</sub>?
  * **Subject:**  O<sub>j</sub>
  * **Answer:** (yes/no)
  * **Location:** L<sub>j</sub>
4. What object is in between O<sub>i</sub> and O<sub>j</sub>?
  * **Subject:**  O<sub>m</sub>
  * **Answer:** O<sub>m</sub>
  * **Location:** L<sub>m</sub>
5. What object is closest to O<sub>i</sub>?
  * **Subject:**  O<sub>i</sub>
  * **Answer:** O<sub>j</sub>
  * **Location:** L<sub>j</sub>
6. What object is farthest from O<sub>i</sub>?
  * **Subject:**  O<sub>i</sub>
  * **Answer:** O<sub>j</sub>
  * **Location:** L<sub>j</sub>
7. What O<sub>i</sub> is closest to O<sub>j</sub>?
  * **Subject:**  O<sub>j</sub>
  * **Answer:** O<sub>i<sub>c</sub></sub>
  * **Location:** L<sub>i<sub>c</sub></sub>
8. What O<sub>i</sub> is farthest from O<sub>j</sub>?
  * **Subject:** O<sub>j</sub>
  * **Answer:** O<sub>i<sub>f</sub></sub>
  * **Location:** L<sub>i<sub>f</sub></sub>
