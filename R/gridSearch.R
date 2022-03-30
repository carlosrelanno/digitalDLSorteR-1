#' @importFrom utils write.csv
#' @importFrom ggplot2 ggplot aes geom_point geom_violin geom_boxplot geom_line geom_abline geom_text geom_hline geom_errorbar geom_bar theme ggtitle element_text xlab ylab scale_color_manual scale_fill_manual scale_x_continuous scale_y_continuous guides guide_legend facet_wrap stat_smooth annotate stat_density_2d element_blank
#' @importFrom reshape2 melt
NULL

gridSearch <- function(
  object,
  params,
  combine = "bulk", ##
  verbose = TRUE, ##
  on.the.fly = FALSE, ##
  subset = 2000,
  prop.test = 1/3,
  models = 100,
  metrics = c("accuracy", "mean_absolute_error",
              "categorical_accuracy", "mean_absolute_percentage_error",
              "kullback_leibler_divergence"),
  save.files = TRUE,
  location = NULL,
  name = NULL
) {

  if (is.null(params)){
    stop("Some parameters must be specified as a list in the 'params' argument")
  }

  if (save.files){
    if (is.null(location)) {
      stop("'location' argument must be specified when saving the data from the models")
    }
    if (is.null(name)) {
      stop("'name' argument must be specified when saving the data from the models")
    }
    # Generate folders
    main.folder <- paste0(location, "/", name)
    if (file.exists(main.folder)){
      message(paste0("A folder with the specified name already exists in the specified location. Creating folder ",
        name, "_1"))
      main.folder <- paste0(main.folder, "_1")
    }
    dir.create(main.folder)
  }

  if (!on.the.fly & "pseudobulk_fun" %in% names(params)){
    warning("The pseudobulk_fun parameter will be removed since there is not on the fly training")
    params <- params[names(params) != "pseudobulk_fun"]
  }
  # Generate combination of parameter's dataframe
  combinations <- expand.grid(params)

  if (models != "all" & models < nrow(combinations)){
    pick <- sample(nrow(combinations), models)
    combinations <- combinations[pick,]
  }
  
  # Generate results table
  test.results <- data.frame(matrix(NA, ncol = length(colnames(combinations)) + length(metrics) + 1,
    nrow = nrow(combinations)))

  train.results <- data.frame(matrix(NA, ncol = length(colnames(combinations)) + 2*(length(metrics) + 1),
    nrow = nrow(combinations)))
  
  metric.names <- c("loss", metrics)
  colnames(test.results) <- c(metric.names, colnames(combinations))

  colnames(train.results) <- c(metric.names, paste0("val_", metric.names), colnames(combinations))

  if (verbose){
    message(paste("Starting grid search with", nrow(combinations), "different models"))
  }

  # Check default parameters
  default.params <- list(
    num_layers = 2,
    num_units = 200,
    num_epochs = 10,
    batch_size = 20,
    dropout_rate = 0.25,
    activation_fun = "relu",
    loss_fun = "kullback_leibler_divergence",
    learning_rate = 0.001,
    pseudobulk_fun = "MeanCPM"
  )
  # Set default parameters if are not defined by user
  defaults <- data.frame(default.params[setdiff(names(default.params), names(params))])
  

  for (i in seq(nrow(combinations))){
    if (verbose){
      message(paste0("Training model ", i, "/", nrow(combinations)))
    }

    if (length(colnames(defaults)) > 0){
      parameters <- cbind(combinations[i,], defaults)
    }
    else {
      parameters <- combinations[i,]
    }

    if (ncol(combinations) == 1){
      names(parameters)[match("combinations[i, ]", names(parameters))] <- names(combinations)
    }

    # message(parameters)

    if (parameters$batch_size > subset * prop.test){
      warning("Batch size is lower than sample size, skipping model")
      test.results[i,] <- c(rep(NA, length(metrics) + 1), combinations[i,])
      next
    }

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
        learning.rate = parameters$learning_rate,

        metrics = metrics,
        on.the.fly = on.the.fly,
        pseudobulk.function = parameters$pseudobulk_fun
    )

    if (save.files){
      model.folder <- paste0(main.folder, "/model_", as.character(i))
      dir.create(model.folder)
      sink(paste0(model.folder, "/samples.txt")); print(model$samples); sink() # // TODO - change this with paste...collapse..
      write.csv(model$train_metrics$metrics, file = paste0(model.folder, "/train_metrics.csv"))
    }

    # // FIXME quiero el nombre de las funciones que usa en los resultados
    test.results[i,] <- c(model$test_metrics, combinations[i,])
    # message(test.results)
    
    train.results[i,] <- c(tail(as.data.frame(model$train_metrics$metrics), 1), combinations[i,])
  }

  if (save.files){
    write.csv(test.results, file = paste0(main.folder, "/grid_search_test_results.csv"))
    write.csv(train.results, file = paste0(main.folder, "/grid_search_train_results.csv"))
  }  
  grid.search(object) <- list(params = params, train.results = train.results, test.results = test.results)

  if (verbose) message("DONE")
  return(object)
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
  learning.rate = 0.001,
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
    optimizer = optimizer_adam(learning_rate = learning.rate),
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
  # 
  shuffling <- sample(seq(nrow(prob.matrix.train)))
  prob.matrix.train <- prob.matrix.train[shuffling,]
  val <- round(0.25 * nrow(prob.matrix.train))

  # Added for validation
  gen.val <- .trainGenerator(
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
    min.index = 1,
    max.index = val,
    threads = threads,
    verbose = verbose
  )

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
    min.index = val+1,
    max.index = nrow(prob.matrix.train),
    threads = threads,
    verbose = verbose
  )

  # history <- suppressWarnings(
  #   model %>% fit_generator(
  #     generator = gen.train,
  #     steps_per_epoch = ceiling(n.train / batch.size),
  #     epochs = num.epochs,
  #     verbose = verbose.model,
  #     view_metrics = view.plot
  #   )

  history <- suppressWarnings(
    model %>% fit_generator(
      generator = gen.train,
      validation_data = gen.val,
      steps_per_epoch = ceiling((nrow(prob.matrix.train)-val)/ batch.size),
      epochs = num.epochs,
      verbose = verbose.model,
      view_metrics = view.plot,
      validation_steps = ceiling(val/ batch.size)
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

saveGridSearchRDS <- function(
  object,
  file
) {
  if (is.null(grid.search(object))){
    stop("This object does not have grid search results data to save")
  }
  saveRDS(grid.search(object), file)
}

loadGridSearchRDS <- function(
  object,
  file
) {
  grid.search(object) <- readRDS(file)
  return(object)
}


gridParamDist <- function(
  object,
  param,
  ...
) {
  if (is.null(grid.search(object))){
    stop("This object does not have grid search results data to analyze")
  }
  if (!(param %in% names(grid.search(object)$params))){
    stop(paste0("This parameter is not available. Available ones are:\n", paste(names(grid.search(object)$params), collapse = " ")))
  }
  plot <- ggplot(grid.search(object)$test.results ,aes(x=factor(.data[[param]])))+ geom_bar(...) + xlab(param)
  return(plot)
}

gridParamSDist <- function(
  object,
  ...
) {
  data <- grid.search(object)$test.results[names(grid.search(object)$params)]
  data <- melt(data)
  plot <- ggplot(data, aes(x=factor(value)))+ geom_bar() + facet_wrap(data$variable, scales = "free_x") + xlab("Value")
  plot <- plot + ylab("Count") + ggtitle(paste("Distribution of parameters for", nrow(grid.search(object)$test.results), "models"))

  return(plot)
}