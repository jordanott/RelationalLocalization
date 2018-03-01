# Dataset Creation

This directory is used to generate relational questions from [Visual Genome](http://visualgenome.org/).

O<sub>i</sub> and O<sub>j</sub> are specific objects in the dataset, where i ≠ j.
### Non-Relational Questions
1. Where is O<sub>i</sub>?
  * **Answer:** O<sub>i</sub>
  * **Location:** L<sub>i</sub>
2. Is it on the left?
  * **Answer:** (yes/no)
  * **Location:** L<sub>i</sub>

### Relational Questions

1. Is  O<sub>i</sub> on the left of O<sub>j</sub>?
  * **Answer:** (yes/no)
  * **Location:** L<sub>i</sub>
2. What object is in between O<sub>i</sub>?
  * **Answer:** O<sub>i</sub>
  * **Location:** L<sub>i</sub>
3. What object is closest to O<sub>j</sub>?
  * **Answer:** O<sub>i</sub>
  * **Location:** L<sub>i</sub>
4. What object is farthest from O<sub>j</sub>?
  * **Answer:** O<sub>i</sub>
  * **Location:** L<sub>i</sub>
5. What O<sub>i</sub> is closest to O<sub>j</sub>?
  * **Answer:** O<sub>i</sub>
  * **Location:** L<sub>i</sub>
6. What O<sub>i</sub> is farthest from O<sub>j</sub>?
  * **Answer:** O<sub>i</sub>
  * **Location:** L<sub>i</sub>
