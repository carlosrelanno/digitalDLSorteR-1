% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gridSearch.R
\name{gridSearch}
\alias{gridSearch}
\title{Perform a grid search hyper parameter tuning of the DNN model}
\usage{
gridSearch(
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
  metrics = c("accuracy", "mean_absolute_error", "categorical_accuracy",
    "mean_absolute_percentage_error", "kullback_leibler_divergence"),
  save.files = FALSE,
  location = NULL,
  name = NULL,
  backup.file = "",
  backup.each = 10
)
}
\arguments{
\item{object}{\code{\linkS4class{DigitalDLSorter}} object with
\code{bulk.simul} slot.}

\item{params}{A named list with different parameters as elements and vectors
of values for each parameter.}

\item{combine}{Type of profiles to be used for training. Can be
\code{'both'}, \code{'single-cell'} or \code{'bulk'} (\code{'bulk'} by
default). For test data, both types of profiles will be used.}

\item{verbose}{Boolean indicating whether to display model progression during
training and model architecture information (\code{TRUE} by default).}

\item{on.the.fly}{Boolean indicating whether data will be generated 'on the
fly' during training (\code{FALSE} by default).}

\item{subset}{Integer indicating how many pseudobulks will be used for the
training and test of each model (2000 by default).}

\item{prop.test}{Proportion of the pseudobulks (\code{subset} parameter) that
will be used for testing each model (1/3 by default).}

\item{prop.val}{Proportion of the training set (1 - prop.test) that will be
used as validation set during training (1/4 by default).}

\item{models}{Number of models to be generated. If it is greater than the
number of possible parameter combinations, or set to \code{all}, all
possible models will be tested.}

\item{metrics}{Vector of metrics used to assess model performance during
training and evaluation (\code{c("accuracy",
  "mean_absolute_error","categorical_accuracy","kullback_leibler_divergence")}
by default). See the
\href{https://keras.rstudio.com/reference/metric_binary_accuracy.html}{keras
documentation} to know available performance metrics.}

\item{save.files}{Boolean indicating if the training history and used
samples for each model are going to be stored in a folder. If \code{TRUE},
the \code{location} and \code{name} arguments must be provided too.}

\item{location}{String indicating the location path for the trained models
data, if \code{save.files = TRUE}.}

\item{name}{String indicating the name of the project to create a folder if
\code{save.files = TRUE}.}

\item{backup.file}{String indicating the name and path of the file to save
the results of the grid search each n models (indicated in
\code{backup.each} parameter). If not specified,no backup will be created.
This is recommended since the process can take a long time.}

\item{backup.each}{Integer indicating every how many models the backup file
should be updated. (10 by default).}
}
\description{
Perform a grid search using a list of parameters to optimize a given metric,
using a \code{\linkS4class{DigitalDLSorter}} object. The results are stored
in the \code{grid.search} slot of the object. In addition, back ups can be
stored on the fly using the \code{backup.file} argument.
}
\details{
The \code{\linkS4class{DigitalDLSorter}} object must contain previously
simulated pseudobulks in the \code{bulk.simul} slot.

The customizable parameters and their defaults are:
num_layers = 2,
num_units = 200,
num_epochs = 10,
batch_size = 20,
dropout_rate = 0.25,
activation_fun = "relu",
loss_fun = "kullback_leibler_divergence",
learning_rate = 0.001,
pseudobulk_fun = "MeanCPM"
}
\examples{
\dontrun{
# Specify the parameter values:
params <- list(
  num_layers = c(2, 4, 6, 8),
  num_units = c(200, 400, 600, 800),
  num_epochs = c(5, 10, 15, 20),
  batch_size = c(20, 50, 100),
  dropout_rate = c(0, 0.25, 0.5),
  activation_fun = c("relu"),
  loss_fun = c("kullback_leibler_divergence", "categorical_crossentropy"),
  learning_rate = c(0.001, 0.005))

# Run the gridSearch function
ddls <- gridSearch(ddls, 
                   params = params, 
                   combine = "both", 
                   subset = 500,
                   prop.test = 0.3,
                   models = 500,
                   location = "save_folder_path",
                   name = "example",
                   metrics = c("accuracy", "mean_absolute_error",
                            "categorical_accuracy", "mean_absolute_percentage_error", 
                            "categorical_crossentropy", "kullback_leibler_divergence"),
                   backup.file = "path",
                   backup.each = 5)
}
}
