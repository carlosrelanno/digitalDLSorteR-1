gridSearch <- function(
  object,
  params,
  combine = "bulk", ##
  verbose = T, ##
  on.the.fly = F, ##
  subset = 2000,
  prop.test = 1/3,
  models = 100,
  metrics = c("accuracy", "mean_absolute_error",
              "categorical_accuracy", "mean_absolute_percentage_error"),
  location,
  name
) {
  if (is.null(location)) {
    stop("'location' argument must be specified")
  }
  # Generate folders
  main.folder <- paste0(location, "/", name)
  if (file.exists(main.folder)){
    print(paste0("A folder with the specified name already exists in the specified location. Creating folder ",
      name, "_1"))
    main.folder <- paste0(main.folder, "_1")
  }
  dir.create(main.folder)

  # Generate combination of parameter's dataframe
  combinations <- expand.grid(params)

  if (models != "all" & models < nrow(combinations)){
    pick <- sample(nrow(combinations), models)
    combinations <- combinations[pick,]
  }
  
  # Generate results table
  results <- data.frame(matrix(NA, ncol = length(colnames(combinations)) + length(metrics) + 1,
    nrow = nrow(combinations)))
  
  colnames(results) = c("loss", metrics, colnames(combinations))
  print(paste("Starting grid search with", nrow(combinations), "different models"))

  for (i in seq(nrow(combinations))){
    print(paste0("Training model ", i, "/", nrow(combinations)))
    parameters <- combinations[i,]
    model <- .trainDigitalDLSorterModel.gridSearch(
        object,
        subset = subset,
        prop.test = prop.test,

        # Parameters
        batch.size = parameters$batch_size,
        num.epochs = parameters$num_epochs,
        num.hidden.layers = parameters$num_layers,
        num.units = rep(parameters$num_units, parameters$num_layers),
        activation.fun = parameters$activation_fun,
        dropout.rate = parameters$dropout_rate,
        loss = parameters$loss_fun,

        metrics = metrics
    )

    model.folder <- paste0(main.folder, "/model_", as.character(i))
    dir.create(model.folder)
    sink(paste0(model.folder, "/samples.txt")); print(model$samples); sink()
    write.csv(model$train_metrics$metrics, file = paste0(model.folder, "/train_metrics.csv"))

    results[i,] <- c(model$test_metrics, combinations[i,])
    print(results)
  }
  write.csv(results, file = paste0(main.folder, "/grid_search_results.csv"))
}

.trainDigitalDLSorterModel.gridSearch <- function(
  object,
  combine = "both",
  batch.size = 64,
  num.epochs = 10,
  num.hidden.layers = 2,
  num.units = c(200, 200),
  activation.fun = "relu",
  dropout.rate = 0.25,
  loss = "kullback_leibler_divergence",
  metrics = metrics,
  scaling = "standarize",
  custom.model = NULL,
  shuffle = FALSE,
  on.the.fly = FALSE,
  pseudobulk.function = "MeanCPM",
  threads = 1,
  view.metrics.plot = TRUE,
  verbose = TRUE,
  subset,
  prop.test
) {
  # check if python dependencies are covered
  .checkPythonDependencies(alert = "error")
  if (!is(object, "DigitalDLSorter")) {
    stop("The provided object is not of DigitalDLSorter class")
  } else if (is.null(prob.cell.types(object))) {
    stop("'prob.cell.types' slot is empty")
  } else if (num.epochs <= 1) {
    stop("'num.epochs' argument must be greater than or equal to 2")
  } else if (batch.size < 10) {
    stop("'batch.size' argument must be greater than or equal to 10")
  } else if ((prop.test < 0) | (prop.test > 1)) {
    stop("'prop.test' argument must be a float between 0 and 1")
  } 
  if (!any(combine %in% c("both", "bulk", "single-cell"))) {
    stop("'combine' argument must be one of the following options: 'both', 'bulk' or 'single-cell'")
  }
  # bulk.simul and single-cell.real/simul must be provided, since we evaluate 
  # our model on both type of samples compulsory
  # check if data provided is correct regarding on the fly training
  if (is.null(single.cell.real(object)) && is.null(single.cell.simul(object))) {
    stop("At least one single-cell slot must be provided ('single.cell.real' ", 
         "or 'single.cell.simul') as trainDigitalDLSorterModel evaluates ", 
         "DNN model on both types of profiles: bulk and single-cell")
  }
  if (!scaling %in% c("standarize", "rescale")) {
    stop("'scaling' argument must be one of the following options: 'standarize', 'rescale'")
  } else {
    if (scaling == "standarize") {
      scaling.fun <- base::scale
    } else if (scaling == "rescale") {
      scaling.fun <- rescale.function
    }
  }
  if (!on.the.fly) {
    if (verbose) message("=== Training and test from stored data was selected")
    if ((combine == "both" && is.null(bulk.simul(object)) ||
         combine == "both" && (is.null(single.cell.real(object)) && 
                               is.null(single.cell.simul(object))))) {
      stop("If 'combine = both' is selected, 'bulk.simul' and at least ",
           "one single cell slot must be provided")
    } else if (combine == "bulk" && is.null(bulk.simul(object))) {
      stop("If 'combine' = bulk is selected, 'bulk.simul' must be provided")
    } else if (is.null(bulk.simul(object, "test"))) {
      stop("trainDigitalDLSorterModel evaluates DNN model on both types of ", 
           "profiles: bulk and single-cell. Therefore, bulk data for test ", 
           "must be provided")
    }
    .pseudobulk.fun <- NULL
  } else {
    if (verbose) message("=== Training and test on the fly was selected")
    if (combine == "both" && (is.null(single.cell.real(object)) && 
                               is.null(single.cell.simul(object)))) {
      stop("If 'combine = both' is selected, at least ",
           "one single cell slot must be provided")
    }
    ## just in case of on.the.fly = TRUE
    if (!pseudobulk.function %in% c("MeanCPM", "AddCPM", "AddRawCount")) {
      stop("'pseudobulk.function' must be one of the following options: 'MeanCPM', 'AddCPM', 'AddRawCount'")
    } else {
      if (pseudobulk.function == "MeanCPM") {
        .pseudobulk.fun <- pseudobulk.fun.mean.cpm
      } else if (pseudobulk.function == "AddCPM") {
        .pseudobulk.fun <- pseudobulk.fun.add.cpm
      } else if (pseudobulk.function == "AddRawCount") {
        .pseudobulk.fun <- pseudobulk.fun.add.raw.counts
      }
    }
  }
  # single-cell must e provided independently of on.the.fly
  if (combine == "single-cell" && (is.null(single.cell.real(object)) && 
                                   is.null(single.cell.simul(object)))) {
    stop("If combine = 'single-cell' is selected, at least ",
         "one single cell slot must be provided")
  }
  if (!is.null(trained.model(object))) {
    warning("'trained.model' slot is not empty. So far, digitalDLSorteR",
            " does not support for multiple trained models, so the current model",
            " will be overwritten\n",
            call. = FALSE, immediate. = TRUE)
  }
  # plots in RStudio during training --> does not work in terminal
  if (view.metrics.plot) view.plot <- "auto"
  else view.plot <- 0
  if (verbose) verbose.model <- 1
  else verbose.model <- 0
  prob.matrix.train <- .targetForDNNSubset(
    object = object, combine = combine, 
    shuffle = TRUE, type.data = "train", 
    fly = on.the.fly, verbose = verbose,
    subset = subset * (1 - prop.test)
  )
  
  prob.matrix.test <- .targetForDNNSubset(
    object = object, combine = "both", 
    shuffle = FALSE, type.data = "test", 
    fly = on.the.fly, verbose = verbose,
    subset = subset * prop.test
  )
  
  n.train <- nrow(prob.matrix.train)
  n.test <- nrow(prob.matrix.test)
  # check if the number of samples is compatible with batch.size
  if (n.train < batch.size) {
    stop(
      paste0("The number of samples used for training (", n.train, ") is too ", 
             "small compared with 'batch.size' (", batch.size, "). Please, ", 
             "increase the number of samples or consider reducing 'batch.size'")
    )
  } 
  if (n.test < batch.size) {
    stop(
      paste0("The number of samples used for test (", n.test, ") is too ", 
             "small compared with 'batch.size' (", batch.size, "). Please, ", 
             "increase the number of samples or consider reducing 'batch.size'")
    )
  }
  if (is.null(custom.model)) {
    if (num.hidden.layers != length(num.units)) {
      stop("The number of hidden layers must be equal to the length of ", 
           "num.units (number of neurons per layer)")
    }
    # check if any argument not provided
    model <- keras_model_sequential(name = "DigitalDLSorter")
    # arbitrary number of hidden layers and neurons
    for (i in seq(num.hidden.layers)) {
      if (i == 1) {
        model <- model %>% layer_dense(
          units = num.units[i], 
          input_shape = nrow(single.cell.real(object)),
          name = paste0("Dense", i)
        )
      } else {
        model <- model %>% layer_dense(
          units = num.units[i], 
          name = paste0("Dense", i)
        )
      }
      model <- model %>% 
        layer_batch_normalization(name = paste0("BatchNormalization", i)) %>%
        layer_activation(activation = activation.fun, 
                         name = paste0("ActivationReLu", i)) %>%
        layer_dropout(rate = dropout.rate, name = paste0("Dropout", i))
    }
    # final layer --> compression and proportions
    model <- model %>% layer_dense(
      units = ncol(prob.cell.types(object, "train") %>% prob.matrix()),
      name = paste0("Dense", i + 1)
    ) %>% layer_batch_normalization(
      name = paste0("BatchNormalization", i + 1)
    ) %>% layer_activation(activation = "softmax", name = "ActivationSoftmax")
  } else {
    # consider more situations where the function fails
    if (!is(custom.model, "keras.engine.sequential.Sequential")) {
      stop("'custom.model' must be a keras.engine.sequential.Sequential object")
    } else if (keras::get_input_shape_at(custom.model$layers[[1]], 1)[[2]] !=
               nrow(single.cell.real(object))) {
      stop("The number of neurons of the first layer must be equal to the ", 
           "number of genes considered by DigitalDLSorter object (", 
           nrow(single.cell.real(object))," in this case)")
    } else if (keras::get_output_shape_at(
        custom.model$layers[[length(custom.model$layers)]], 1
      )[[2]] != ncol(prob.cell.types(object, "train") %>% prob.matrix())) {
      stop("The number of neurons of the last layer must be equal to the ", 
           "number of cell types considered by DigitalDLSorter object (", 
           ncol(prob.cell.types(object, "train") %>% prob.matrix()), 
           " in this case)")
    } else if (!grepl("'activation': 'softmax'", keras::get_config(custom.model))) {
      stop("In order to get proportions as output, the activation function of the ",
           "last hidden layer must be 'softmax'")
    }
    model <- custom.model
  }
  if (verbose) summary(model)
  # allow set optimizer?
  model %>% compile(
    loss = loss,
    optimizer = optimizer_adam(),
    metrics = metrics
  )
  # pattern to set simulated and real cells
  if (!is.null(single.cell.simul(object))) {
    suffix.names <- unique(colData(single.cell.simul(object))$suffix)
  } else {
    suffix.names <- "_Simul"
  }
  pattern <- suffix.names
  # set if samples will be generated on the fly
  if (!on.the.fly) {
    .dataForDNN <- .dataForDNN.file
  } else {
    .dataForDNN <- .dataForDNN.onFly
  }
  if (verbose) 
    message(paste("\n=== Training DNN with", n.train, "samples:\n"))
  gen.train <- .trainGenerator(
    object = object, 
    funGen = .dataForDNN,
    prob.matrix = prob.matrix.train,
    type.data = "train",
    fun.pseudobulk = .pseudobulk.fun,
    scaling = scaling.fun,
    batch.size = batch.size,
    combine = combine,
    shuffle = shuffle,
    pattern = pattern,
    min.index = NULL,
    max.index = NULL,
    threads = threads,
    verbose = verbose
  )
  history <- suppressWarnings(
    model %>% fit_generator(
      generator = gen.train,
      steps_per_epoch = ceiling(n.train / batch.size),
      epochs = num.epochs,
      verbose = verbose.model,
      view_metrics = view.plot
    )
  )
  # }
  if (verbose)
    message(paste0("\n=== Evaluating DNN in test data (", n.test, " samples)"))

  # evaluation of the model: set by default, no options?
  gen.test <- .predictGenerator(
    object,
    funGen = .dataForDNN,
    target = TRUE,
    prob.matrix = prob.matrix.test,
    fun.pseudobulk = .pseudobulk.fun,
    scaling = scaling.fun,
    batch.size = batch.size,
    pattern = pattern,
    threads = threads,
    verbose = verbose
  )
  test.eval <- suppressWarnings(
    model %>% evaluate_generator(
      generator = gen.test,
      steps = ceiling(n.test / batch.size)
    )
  )
  # prediction of test samples
  if (verbose) {
    message(paste0("   - ", names(test.eval), ": ", lapply(test.eval, round, 4),
                   collapse = "\n"))
  #   message(paste("\n=== Generating prediction results using test data\n"))
  }
  # gen.predict <- .predictGenerator(
  #   object,
  #   funGen = .dataForDNN,
  #   target = FALSE,
  #   prob.matrix = prob.matrix.test,
  #   fun.pseudobulk = .pseudobulk.fun,
  #   scaling = scaling.fun,
  #   batch.size = batch.size,
  #   pattern = pattern,
  #   threads = threads,
  #   verbose = verbose
  # )
  # predict.results <- suppressWarnings(
  #   model %>% predict_generator(
  #     generator = gen.predict,
  #     steps = ceiling(n.test / batch.size),
  #     verbose = verbose.model
  #   )
  # )
  # rownames(predict.results) <- rownames(prob.matrix.test)
  # colnames(predict.results) <- colnames(prob.matrix.test)
  
  # network.object <- new(
  #   Class = "DigitalDLSorterDNN",
  #   model = model,
  #   training.history = history,
  #   test.metrics = test.eval,
  #   test.pred = predict.results,
  #   cell.types = colnames(prob.matrix.test),
  #   features = rownames(single.cell.real(object))
  # )
  # trained.model(object) <- network.object
  return(list(test_metrics = test.eval, train_metrics = history, samples = list(train = rownames(prob.matrix.train),
  test = rownames(prob.matrix.test))))
  if (verbose) message("DONE")
  # return(object)
}


.targetForDNNSubset <- function(
  object, 
  combine,
  type.data,
  fly,
  shuffle,
  subset,
  verbose
) {
  if (combine == "both") {
    tpsm <- matrix(
      unlist(sapply(
        X = names(prob.cell.types(object, type.data) %>% set.list()),
        FUN = function (x, l) {
          v <- rep(0, length(l))
          names(v) <- names(l)
          v[x] <- 1
          return(rep(v, length(l[[x]])))
        }, 
        l = prob.cell.types(object, type.data) %>% set.list()
      )), 
      ncol = length(prob.cell.types(object, type.data) %>% set.list()), 
      byrow = TRUE,
      dimnames = list(unlist(prob.cell.types(object, type.data) %>% set.list()),
                      names(prob.cell.types(object, type.data) %>% set.list()))
    )
    allCellTypes <- colnames(prob.cell.types(object, type.data) %>% prob.matrix())
    if (!all(allCellTypes %in% colnames(tpsm))) {
      lackTypes <- allCellTypes[!allCellTypes %in% colnames(tpsm)]
      lackMatrix <- matrix(
        0, ncol = length(lackTypes), nrow = nrow(tpsm), 
        dimnames = list(rownames(tpsm), lackTypes)
      )
      tpsm <- cbind(tpsm, lackMatrix)
    }
    tpsm <- tpsm[, colnames(prob.cell.types(object, type.data) %>% 
                              prob.matrix())]
    
    if (fly) {
      probs.matrix <- rbind(
        tpsm, prob.cell.types(object, type.data) %>% prob.matrix() / 100
      )  
      rownames(probs.matrix) <- c(
        rownames(tpsm), 
        rownames(prob.cell.types(object, type.data) %>% prob.matrix())
      )
    } else {
      tpsm <- tpsm[sample(nrow(tpsm)), ]
      probs.matrix <- prob.cell.types(object, type.data)@prob.matrix[
        colnames(bulk.simul(object, type.data)), ] / 100
      if (nrow(tpsm) > nrow(probs.matrix)) {
        probs.matrix <- .mergePropsSort(m.small = probs.matrix, m.big = tpsm)
      } else if (nrow(tpsm) <= nrow(probs.matrix)) {
        probs.matrix <- .mergePropsSort(m.small = tpsm, m.big = probs.matrix)
      }
    }
  } else if (combine == "bulk") {
    if (verbose) message("    Using only simulated bulk samples\n")
    if (fly) {
      probs.matrix <- prob.cell.types(object, type.data) %>% prob.matrix() / 100
    } else {
      probs.matrix <- prob.cell.types(object, type.data)@prob.matrix[
        colnames(bulk.simul(object, type.data)), ] / 100
    }
  } else if (combine == "single-cell") {
    if (verbose) message("    Using only single-cell samples\n")
    probs.matrix <- matrix(
      unlist(sapply(
        X = names(prob.cell.types(object, type.data) %>% set.list()),
        FUN = function (x, l) {
          v <- rep(0, length(l))
          names(v) <- names(l)
          v[x] <- 1
          return(rep(v, length(l[[x]])))
        }, l = prob.cell.types(object, type.data) %>% set.list()
      )), ncol = length(prob.cell.types(object, type.data) %>% set.list()), 
      byrow = TRUE,
      dimnames = list(unlist(prob.cell.types(object, type.data) %>% set.list()),
                      names(prob.cell.types(object, type.data) %>% set.list()))
    )
    allCellTypes <- colnames(prob.cell.types(object, type.data) %>% prob.matrix())
    if (!any(allCellTypes %in% colnames(probs.matrix))) {
      lackTypes <- allCellTypes[!allCellTypes %in% colnames(probs.matrix)]
      lackMatrix <- matrix(
        0, ncol = length(lackTypes), nrow = nrow(probs.matrix), 
        dimnames = list(rownames(probs.matrix), lackTypes)
      )
      probs.matrix <- cbind(probs.matrix, lackMatrix)
    }
    probs.matrix <- probs.matrix[
      , colnames(prob.cell.types(object, type.data) %>% prob.matrix())]
  }
  # Subset samples 
  selected_samples <- sample(nrow(probs.matrix), subset) 
  probs.matrix <- probs.matrix[selected_samples,] ## DEVOLVER LAS MUESTRAS QUE USA???

  # shuffle only if train on the fly
  if (shuffle && !fly) {
    return(probs.matrix[sample(nrow(probs.matrix)), ])
  } else {
    return(probs.matrix)
  }
}
