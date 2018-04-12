# Relational Localization

## Results
| | RN (Ans, Loc) | Baseline (Ans, Loc) | RN (Ans) | Baseline (Ans)
| --- | --- | --- | --- | --- |
| Non-relational question | 98.93% | 78.89% | **99.17%** | 77.87% |
| Relational question | **73.26%** | 42.82% | 71.82% | 45.71% |
| Overall Acc | **88.69%** | 64.43% | 88.19% | 65.03% |
| Non-relational IoU | **0.61** | 0.11 | ----------- | ----------- |
| Relational IoU | **0.17** | 0.09 | ----------- | ----------- |
| Overall IoU | **0.43** | 0.10 | ----------- | ----------- |

### To Do
- [x] SSD for real time detection, streamed from camera  
- [x] Modify RN for localization
- [x] Create dataset from real images
  - [x] Relational questions
  - [x] Non-relational questions
  - [x] Bounding box coordinates (x,y,w,h)
  - [x] Visualize dataset examples
- [x] Experiments
  - [x] CNN without RN
    - [x] Non-relational questions
    - [x] Non-relational & relational questions
  - [x] CNN with RN
    - [x] Non-relational questions
    - [x] Non-relational & relational questions
  - [ ] Question embedding options
    - [ ] Preset one-hot questions (no RNN)
    - [ ] With RNN to process questions
- [x] Visualize bounding box results on images
