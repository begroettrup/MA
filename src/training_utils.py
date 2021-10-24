from interfaces.test_results import TestResults

def train_and_validate(model, validation_set,
  validation_metric="Loss",
  bigger_is_better=False,
  add_metrics=[], early_stopping=None):
  """
  Trains a model on a validation set and returns the best model and validation
  result. The model will be trained until its configured epoch unless early
  stopping is used and triggered.

  Args:
    early_stopping: Number of epochs that the result did not improve.
    bigger_is_better: If set bigger values of the validation metric will be
      treated as being better. Default is false, i.e. lower is better.
    add_metrics: Additional metrics besides the validation metric to measure
      and print.
  """
  best_epoch = None

  epochs = model.get_parameter("epochs")

  metrics = ["Loss"] + add_metrics

  if bigger_is_better:
    is_better = lambda new, old: new > old
  else:
    is_better = lambda new, old: new < old

  def make_test(model):
    model_test_results = TestResults()
    model_test_results.set_parameter("model", model)
    model_test_results.set_parameter("testset", validation_set)
    model_test_results.set_parameter("metrics", metrics)

    return model_test_results

  for epoch in range(1, epochs+1):
    model.set_parameter("epochs", epoch)
    model.ensure_available()

    metric_results = make_test(model).reproduction()
    val_loss = metric_results[0]

    metrics_string = ""

    for mname, mresult in zip(metrics, metric_results):
      if metrics_string != "":
        metrics_string += "; "
      metrics_string += mname + ": " + str(mresult)

    print("Epoch " + str(epoch) + ": " + metrics_string)

    if best_epoch is None or is_better(val_loss, best_loss):
      best_epoch = epoch
      best_loss = val_loss
      no_improvements = 0
    else:
      no_improvements += 1

    if no_improvements == early_stopping:
      print("Stopped training due to early stopping.")
      break

  print("Training done, best epoch was", best_epoch)
  model.set_parameter("epochs", best_epoch)

  return model, make_test(model)
