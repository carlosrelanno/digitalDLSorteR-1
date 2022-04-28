#' @importFrom utils write.csv
#' @importFrom ggplot2 ggplot aes geom_point geom_violin geom_boxplot geom_histogram geom_line geom_abline geom_text geom_hline geom_errorbar geom_bar theme ggtitle element_text xlab ylab scale_color_manual scale_fill_manual scale_x_continuous scale_y_continuous guides guide_legend facet_wrap stat_smooth annotate stat_density_2d element_blank
#' @importFrom reshape2 melt
#' @importFrom ComplexHeatmap Heatmap draw HeatmapAnnotation
NULL

################################################################################
####################### Grid search using the DNN model ########################
################################################################################

#' Perform a grid search hyper parameter tuning of the DNN model
#'
#' Perform a grid search using a list of parameters to optimize a given metric,
#' using a \code{\linkS4class{DigitalDLSorter}} object. The results are stored
#' in the \code{grid.search} slot of the object. In addition, back ups can be
#' stored on the fly using the \code{backup.file} argument.
#'
#' The \code{\linkS4class{DigitalDLSorter}} object must contain previously
#' simulated pseudobulks in the \code{bulk.simul} slot.
#' 
#' The customizable parameters and their defaults are:
#'    num_layers = 2,
#'    num_units = 200,
#'    num_epochs = 10,
#'    batch_size = 20,
#'    dropout_rate = 0.25,
#'    activation_fun = "relu",
#'    loss_fun = "kullback_leibler_divergence",
#'    learning_rate = 0.001,
#'    pseudobulk_fun = "MeanCPM"
#'
#' @param object  \code{\linkS4class{DigitalDLSorter}} object with
#'   \code{bulk.simul} slot.
#' @param params  A named list with different parameters as elements and vectors
#'   of values for each parameter.
#' @param combine Type of profiles to be used for training. Can be
#'   \code{'both'}, \code{'single-cell'} or \code{'bulk'} (\code{'bulk'} by
#'   default). For test data, both types of profiles will be used.
#' @param verbose Boolean indicating whether to display model progression during
#'   training and model architecture information (\code{TRUE} by default).
#' @param on.the.fly  Boolean indicating whether data will be generated 'on the
#'   fly' during training (\code{FALSE} by default).
#' @param subset  Integer indicating how many pseudobulks will be used for the
#'   training and test of each model (2000 by default).
#' @param prop.test Proportion of the pseudobulks (\code{subset} parameter) that
#'   will be used for testing each model (1/3 by default).
#' @param prop.val Proportion of the training set (1 - prop.test) that will be
#'   used as validation set during training (1/4 by default).
#' @param models Number of models to be generated. If it is greater than the
#'   number of possible parameter combinations, or set to \code{all}, all
#'   possible models will be tested.
#' @param metrics Vector of metrics used to assess model performance during
#'   training and evaluation (\code{c("accuracy",
#'   "mean_absolute_error","categorical_accuracy","kullback_leibler_divergence")}
#'    by default). See the
#'   \href{https://keras.rstudio.com/reference/metric_binary_accuracy.html}{keras
#'    documentation} to know available performance metrics.
#' @param save.files  Boolean indicating if the training history and used
#'   samples for each model are going to be stored in a folder. If \code{TRUE},
#'   the \code{location} and \code{name} arguments must be provided too.
#' @param location  String indicating the location path for the trained models
#'   data, if \code{save.files = TRUE}.
#' @param name  String indicating the name of the project to create a folder if
#'   \code{save.files = TRUE}.
#' @param backup.file  String indicating the name and path of the file to save
#'   the results of the grid search each n models (indicated in
#'   \code{backup.each} parameter). If not specified,no backup will be created.
#'   This is recommended since the process can take a long time.
#' @param backup.each Integer indicating every how many models the backup file
#'   should be updated. (10 by default).
#'   
#' @examples
#' \dontrun{
#' # Specify the parameter values:
#' params <- list(
#'   num_layers = c(2, 4, 6, 8),
#'   num_units = c(200, 400, 600, 800),
#'   num_epochs = c(5, 10, 15, 20),
#'   batch_size = c(20, 50, 100),
#'   dropout_rate = c(0, 0.25, 0.5),
#'   activation_fun = c("relu"),
#'   loss_fun = c("kullback_leibler_divergence", "categorical_crossentropy"),
#'   learning_rate = c(0.001, 0.005))
#' 
#' # Run the gridSearch function
#' ddls <- gridSearch(ddls, 
#'                    params = params, 
#'                    combine = "both", 
#'                    subset = 500,
#'                    prop.test = 0.3,
#'                    models = 500,
#'                    location = "save_folder_path",
#'                    name = "example",
#'                    metrics = c("accuracy", "mean_absolute_error",
#'                             "categorical_accuracy", "mean_absolute_percentage_error", 
#'                             "categorical_crossentropy", "kullback_leibler_divergence"),
#'                    backup.file = "path",
#'                    backup.each = 5)
#' }
gridSearch <- function( 
  object,
  params = NULL,
  combine = "bulk",
  verbose = TRUE,
  view.metrics.plot = TRUE, 
  on.the.fly = FALSE, 
  subset = 2000,
  prop.test = 1/3,
  prop.val = 1/4,
  models = 100,
  metrics = c("accuracy", "mean_absolute_error",
              "categorical_accuracy", "mean_absolute_percentage_error",
              "kullback_leibler_divergence"),
  save.files = FALSE,
  location = NULL,
  name = NULL,
  backup.file = "",
  backup.each = 10
) {
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

  if (is.null(params)){
    stop("Some parameters must be specified as a list in the 'params' argument")
  }
  else if (class(params) != "list"){
    stop("'params' argument must be a list. Check documentation for more information.")
  }
  else if (is.null(names(params))){
    stop("'params' argument must be a named list. Check documentation for more information.")
  }
  else if (!all(names(params) %in% names(default.params))){
    stop("Parameter names do not match with the available ones. Check documentation for more information.")
  }
  if (class(object) != "DigitalDLSorter"){
    stop("'object' argument must be of class DigitalDLSorter")
  }
  if (is.null(bulk.simul(object))){
    stop("'bulk.simul' slot is empty")
  }

  if (!on.the.fly) {
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
  }}

  if (on.the.fly & !("pseudobulk_fun" %in% names(params))){
    stop("'on.the.fly' option selected but no given parameters for on the fly bulk generation functions")
  }

  if (prop.test > 1 | prop.test < 0){
    stop("'prop.test' must be a number between 0 and 1")
  }
   if (prop.val > 1 | prop.val < 0){
    stop("'prop.val' must be a number between 0 and 1")
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
      warning("Sample size is lower than batch size, skipping model")
      test.results[i,] <- c(rep(NA, length(metrics) + 1), combinations[i,])
      next
    }

    model <- .trainDigitalDLSorterModel.gridSearch(
        object,
        subset = subset,
        prop.test = prop.test,
        prop.val = prop.val,

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
        pseudobulk.function = parameters$pseudobulk_fun,
        view.metrics.plot = view.metrics.plot
    )

    if (save.files){
      model.folder <- paste0(main.folder, "/model_", as.character(i))
      dir.create(model.folder)
      sink(paste0(model.folder, "/samples.txt")); print(model$samples); sink() # // TODO - change this with paste...collapse..
      write.csv(model$train_metrics$metrics, file = paste0(model.folder, "/train_metrics.csv"))
    }

    test.results[i,] <- c(model$test_metrics, combinations[i,])
    # message(test.results)
    
    train.results[i,] <- c(tail(as.data.frame(model$train_metrics$metrics), 1), combinations[i,])

    if (backup.file != "" & i%%backup.each == 0){
      saveRDS(list(params = params, train.results = na.omit(train.results), test.results = na.omit(test.results)), backup.file)
    }
    keras::k_clear_session() ### !!!!!!!!!!!!!!!!!!!!!! https://github.com/rstudio/keras/issues/339
    message("using k_clear_session")
  }

  if (save.files){
    write.csv(test.results, file = paste0(main.folder, "/grid_search_test_results.csv"))
    write.csv(train.results, file = paste0(main.folder, "/grid_search_train_results.csv"))
  }

  # Get the name of the categorical parameters
  string.params <- names(params)[unlist(lapply(params, is.character))] 
  string.params <- string.params[string.params %in% colnames(test.results)]

  for (p in string.params){
    train.results[p] <- lapply(train.results[p], function(x) (params[[p]][x]))
    test.results[p] <- lapply(test.results[p], function(x) (params[[p]][x]))
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
  prop.val = 0.25,
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
    model <- .buildModel(
      object,
      num.hidden.layers = num.hidden.layers,
      num.units = num.units,
      activation.fun = activation.fun,
      dropout.rate = dropout.rate
    )
    
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
  val <- round(prop.val * nrow(prob.matrix.train))

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
  if (subset < nrow(probs.matrix)){
    selected_samples <- sample(nrow(probs.matrix), subset)
    probs.matrix <- probs.matrix[selected_samples,] ## DEVOLVER LAS MUESTRAS QUE USA???
  }
  else{
    warning(paste("'subset' parameter is greater than then number of pseudobulks. All pseudobulks are being used for", type.data))
  } 
  

  # shuffle only if train on the fly
  if (shuffle && !fly) {
    return(probs.matrix[sample(nrow(probs.matrix)), ])
  } else {
    return(probs.matrix)
  }
}


#' Save the grid.search slot in a RDS file
#'
#' Write a RDS file containing the \code{grid.search} slot of a
#' \code{\linkS4class{DigitalDLSorter}} object.
#' @param object \code{\linkS4class{DigitalDLSorter}} object with
#'   \code{grid.search} slot.
#' @param file A string indicating the path and file name to store the data.
saveGridSearchRDS <- function(
  object,
  file
) {
  if (is.null(grid.search(object))){
    stop("This object does not have grid search results data to save")
  }
  saveRDS(grid.search(object), file)
}


#' Load the grid.search slot from a RDS file
#'
#' Load a RDS file containing the \code{grid.search} slot of a
#' \code{\linkS4class{DigitalDLSorter}} object.
#' @param object \code{\linkS4class{DigitalDLSorter}}
#' @param file A string indicating the path and file name to load the data from.
loadGridSearchRDS <- function(
  object,
  file
) {
  grid.search(object) <- readRDS(file)
  return(object)
}


#' Histogram of the distribution of a given parameter
#'
#' Display the distribution of a given parameter between all generated models.
#'
#' @param object \code{\linkS4class{DigitalDLSorter}} object with
#'   \code{grid.search} slot.
#' @param param The parameter to plot. The available parameters can be known
#'   using \code{params()}
#' @param theme \pkg{ggplot2} theme.
#' @param ... 
#' 
#' @example 
#' \dontrun{
#' gridParamDist(ddls, param = "num_layers")
#' }
gridParamDist <- function(
  object,
  param,
  theme = NULL,
  ...
) {
  if (is.null(grid.search(object))){
    stop("This object does not have grid search results data to analyze")
  }
  if (!(param %in% names(grid.search(object)$params))){
    stop(paste0("This parameter is not available. Available ones are:\n", paste(names(grid.search(object)$params), collapse = " ")))
  }
  plot <- ggplot(grid.search(object)$test.results ,aes(x=factor(.data[[param]])))+ geom_bar(...) + xlab(param)
  plot <- plot + ylab("Count")
  plot <- plot + DigitalDLSorterTheme() + theme
  return(plot)
}

#' Histograms of the distribution of parameters
#'
#' Display the distribution of all parameters between all generated models.
#'
#' @param object \code{\linkS4class{DigitalDLSorter}} object with
#'   \code{grid.search} slot.
#' @param theme \pkg{ggplot2} theme.
#' @param ... 
gridParamSDist <- function(
  object,
  theme = NULL,
  ...
) {
  data <- grid.search(object)$test.results[names(grid.search(object)$params)]
  data <- reshape2::melt(data, id.vars=NULL)
  plot <- ggplot(data, aes(x=factor(value)))+ geom_bar() + facet_wrap(data$variable, scales = "free_x") + xlab("Value")
  plot <- plot + ylab("Count") + ggtitle(paste("Distribution of parameters for", nrow(grid.search(object)$test.results), "models"))
  plot <- plot + DigitalDLSorterTheme() + theme

  return(plot)
}

#' Histogram of the distribution of a given metric
#'
#' Display the distribution of a given metric between all generated models.
#'
#' @param object \code{\linkS4class{DigitalDLSorter}} object with
#'   \code{grid.search} slot.
#' @param metric The metric to plot. The available metrics can be known using
#'   \code{metrics()}.
#' @param set The data set to use. Can be set to \code{test.results} (evaluation
#'   using the test set) or \code{train.results} (data about the last training
#'   epoch, with train and validation metrics).
#' @param normalize Boolean indicating if the metric is going to be normalized.
#' @param theme \pkg{ggplot2} theme.
gridMetricDist <- function(
  object,
  metric = "loss",
  set = "test.results",
  normalize = TRUE,
  theme = NULL,
  ...
) {
   if (is.null(grid.search(object))){
    stop("This object does not have grid search results data to analyze")
  }

  metrics <- setdiff(colnames(grid.search(object)[[set]]), names(grid.search(object)$params))
  if (!(metric %in% metrics)){
    stop(paste0("This metric is not available. Available ones are:\n", paste(metrics, collapse = " ")))
  }

  if (normalize){
    metric.res <- as.data.frame(apply(grid.search(object)[[set]][,metrics],2,.normdata2))
  }
  else {metric.res <- grid.search(object)[[set]][,metrics]}

  plot <- ggplot(metric.res ,aes(x=.data[[metric]]))+ geom_histogram() + xlab(metric)
  plot <- plot + ylab("Count")
  plot <- plot + DigitalDLSorterTheme() + theme
  return(plot)
}

#' Histograms of the distribution of metrics
#'
#' Display the distribution of all metrics between all generated models.
#'
#' @param object \code{\linkS4class{DigitalDLSorter}} object with
#'   \code{grid.search} slot.
#' @param set The data set to use. Can be set to \code{test.results} (evaluation
#'   using the test set) or \code{train.results} (data about the last training
#'   epoch, with train and validation metrics).
#' @param theme \pkg{ggplot2} theme.
#' @param normalize Boolean indicating if the metrics are going to be
#'   normalized.
#' @param ...
gridMetricSDist <- function(
  object,
  set = "test.results",
  normalize = TRUE,
  theme = NULL,
  ...
) {
   if (is.null(grid.search(object))){
    stop("This object does not have grid search results data to analyze")
  }

  metrics <- setdiff(colnames(grid.search(object)[[set]]), names(grid.search(object)$params))
  
  if (normalize){
    metric.res <- as.data.frame(apply(grid.search(object)[[set]][,metrics],2,.normdata2))
  }
  else {metric.res <- grid.search(object)[[set]][,metrics]}

  metric.res <- reshape2::melt(metric.res, id.vars=NULL)

  plot <- ggplot(metric.res, aes(x=value))+ ggplot2::geom_histogram() + facet_wrap(metric.res$variable, scales = "free_x") + xlab("Value")
  plot <- plot + ylab("Count") + ggtitle(paste("Distribution of metrics for", nrow(grid.search(object)$test.results), "models"))
  plot <- plot + DigitalDLSorterTheme() + theme
  return(plot)
}

#' Complex heatmap with the best models and parameters
#'
#' Display the best models according to a given metric and visualize their
#' parameters.
#'
#' @param object \code{\linkS4class{DigitalDLSorter}} object with
#'   \code{grid.search} slot.
#' @param sort.metric The metric that is going to be used to sort the models. To
#'   see available metrics use \code{metrics()}.
#' @param decreasing Boolean indicating if the sort order is decreasing. Useful
#'   if sorting by accuracy.
#' @param n.models Integer indicating the number of models to be displayed.
gridMap <- function(
  # // TODO Cambiar colores, coloracón por ranking de modelos, añadir numeros de metricas
  object,
  sort.metric = "kullback_leibler_divergence",
  decreasing = FALSE,
  n.models = 20
) {
  metrics <- setdiff(colnames(grid.search(object)$test.results), names(grid.search(object)$params))
  metrics <- metrics[metrics != "loss"]
  metric.res <- as.data.frame(apply(grid.search(object)$test.results[,metrics],2,.normdata2))

  models <- order(grid.search(object)$test.results[,sort.metric], decreasing=decreasing)[1:n.models]
  paramsAnnotDf <- as.data.frame(apply(grid.search(object)$test.results[,names(grid.search(object)$params)],2,function (x) {return(factor(x))}))
  a = ComplexHeatmap::HeatmapAnnotation(df = paramsAnnotDf[models,], which = "row")

  string.params <- names(grid.search(object)$params)[unlist(lapply(grid.search(object)$params, is.character))] 
  paramsDF <- grid.search(object)$test.results[models, names(grid.search(object)$params)]

  for (p in string.params) {paramsDF[,p] <- factor(paramsDF[,p])}

  hm <- ComplexHeatmap::Heatmap(
    as.matrix(metric.res[models,]),
    cluster_rows = hclust(cluster::daisy(paramsDF, metric="gower")),
    show_row_names = T) + a
  ComplexHeatmap::draw(hm)
}

bestModels <- function(
  object,
  metric = "loss",
  decreasing = FALSE,
  n.models = 5
) {
    models <- grid.search(object)$test.results[order(grid.search(object)$test.results[,metric], decreasing = decreasing),][1:n.models,]
    return(models)
}

gridCorr <- function(
  object
) {
  parameters <- grid.search(object)$test.results[names(grid.search(object)$params)] %>% dplyr::mutate_all(as.factor)
  metrics <- setdiff(colnames(grid.search(object)$test.results), names(grid.search(object)$params))
  metrics <- metrics[metrics != "loss"]
  metric.res <- as.data.frame(apply(grid.search(object)$test.results[,metrics], 2, .normdata2))
  selected <- apply(parameters, 2, function (x) (var(x) != 0 & !is.na(var(x))))
  parameters <- parameters[,selected]

  data <- cbind(metric.res, parameters)
  maeCorr <- polycor::hetcor(data)
  hm <- ComplexHeatmap::Heatmap(maeCorr$correlations,cluster_rows = F,cluster_columns = F)
  return(hm)
}

boxplotGrid <- function(
  object,
  param,
  metric = "loss",
  jitter.by = NULL,
  facet.by = NULL,
  title = NULL,
  quantile.cut = 0.95,
  theme = NULL
) {
  if (!is.null(jitter.by)){
    jitter <- ggplot2::geom_jitter(size=2, aes(color=factor(.data[[jitter.by]]))) 
  }
  else {jitter = NULL}

  if (!is.null(facet.by)){
    facet <- ggplot2::facet_grid(grid.search(object)$test.results[[facet.by]])
  }
  else {facet = NULL}

  if (is.null(title)){
    title <- paste(stringr::str_to_title(metric), "by", param)
  }

  plot <- ggplot(
    grid.search(object)$test.results,aes(x=factor(.data[[param]]),y=.data[[metric]])) +
    geom_boxplot(outlier.shape = NA) + 
    xlab(param) + 
    jitter + 
    facet + 
    ggplot2::labs(color= jitter.by, title=title) +
    ggplot2::coord_cartesian(y = c(0, quantile(grid.search(object)$test.results[,metric], quantile.cut)))
  
  plot <- plot + DigitalDLSorterTheme() + theme
  return(plot)
}

paramsBoxplot <- function(
  object,
  metric = "loss",
  quantile.cut = 0.95,
  facet.by = NULL 
  # jitter = FALSE mejor no
) {
  plots <- lapply(params(object), 
    function (p) (boxplotGrid(ddls, 
      metric = metric, 
      param = p, 
      title = as.character(p), 
      quantile.cut = quantile.cut)))
  return(gridExtra::grid.arrange(grobs=plots)) 
    #top=grid::textGrob(stringr::str_to_title(metric), gp=gpar(fontsize=18))))
}

# .jitter.paramsBoxplot <- function(
#   jitter, 
#   p
# ){
#   if (jitter){return(p)}
#   else {return(NULL)}
# }

.normdata2 <- function(
  x
) {
  return((x-min(x))/(max(x)-min(x)))
}

params <- function(
  object
) {
  return(names(grid.search(object)$params))
}

metrics <- function(
  object
) {
  metrics <- setdiff(colnames(grid.search(object)$test.results), names(grid.search(object)$params))
  return(metrics)
  }

append.grid <- function(
  object,
  grid.search.list
) {
  if (!all(unlist(grid.search(object)$params) == unlist(grid.search.list$params))){
    stop("The paremeters of these experiments are not the same")
  }

  if (!all(colnames(grid.search(object)$train.results) == colnames(grid.search.list$train.results))){
    stop("The columns in the train set do not match")
  }

  if (!all(colnames(grid.search(object)$test.results) == colnames(grid.search.list$test.results))){
    stop("The columns in the test set do not match")
  }
  # Train
  grid.search(object)$train.results <- rbind(grid.search(object)$train.results, grid.search.list$train.results)
  # Test
  grid.search(object)$test.results <- rbind(grid.search(object)$test.results, grid.search.list$test.results)
  return(object)
}