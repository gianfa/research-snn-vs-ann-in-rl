# Things noticeable during hand tuning

## 1. At first tune the minimum I may be beneficial.

    It may be done turning off the recurrent connections.

## 2. Remember that Readout_LIF weights are free to move between -inf and inf

## 3. Maybe a not-mutual chain architecture is more able to diffrentiate

## 4. a larger radius shows a more sustained activity. `~ reservoir/2``

## 5. Adam, `lr ~1e-2`.

## 6. increasing the `input_to_reservoir` from I, the acc increases

## 7. Reservoir weight_dist "uniform", with approriate connection gain (which is now spread), the acc increases

## 8. Loss scope must be appropriately large

In other words if the update frequency is too large, there will be no learning (no trend in loss).

<img src='./imgs/loss_not_learning.png'>