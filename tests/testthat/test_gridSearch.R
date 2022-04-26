context("Grid search hyperparameter tuning: gridSearch.R")

skip_if_not(.checkPythonDependencies(alert = "none"))

# to make compatible with any computer --> disable eager execution
tensorflow::tf$compat$v1$disable_eager_execution()

################################################################################
############################ gridSearch function ###############################
################################################################################

set.seed(123)
sce <- SingleCellExperiment(
  matrix(
    stats::rpois(100, lambda = 5), nrow = 40, ncol = 30, 
    dimnames = list(paste0("Gene", seq(40)), paste0("RHC", seq(30)))
  ),
  colData = data.frame(
    Cell_ID = paste0("RHC", seq(30)),
    Cell_Type = sample(x = paste0("CellType", seq(4)), size = 30, replace = TRUE)
  ),
  rowData = data.frame(
    Gene_ID = paste0("Gene", seq(40))
  )
)
DDLS <- loadSCProfiles(
  single.cell.data = sce,
  cell.ID.column = "Cell_ID",
  gene.ID.column = "Gene_ID"
)

params <- list(
  num_layers = c(2, 4),
  num_units = c(200),
  num_epochs = c(3),
  batch_size = c(20)) 

test_that(
  "Wrong object: lack of specific data", 
  {
    # object without bulk.simul slot
    expect_error(
    gridSearch(object = DDLS, params = params), 
      regexp = "'bulk.simul' slot is empty")
  }
)
test_that(
    "Wrong parameters",
    {
        # params is not a list
        expect_error(
            gridSearch(object = DDLS, params = "bad"),
            regexp = "'params' argument must be a list. Check documentation for more information."
        )
        # params is not an empty list
        expect_error(
            gridSearch(object = DDLS, params = list()),
            regexp = "'params' argument must be a named list. Check documentation for more information."
        )
        expect_error(
            gridSearch(object = DDLS, params = list(bad.param = c(1, 2))),
            regexp = "Parameter names do not match with the available ones. Check documentation for more information."
        )
    }
)