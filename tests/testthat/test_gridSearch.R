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
        gridSearch(object = "a", params = params),
        regexp = "'object' argument must be of class DigitalDLSorter"
    )
    expect_error(
    gridSearch(object = DDLS, params = params), 
      regexp = "'bulk.simul' slot is empty"
    )
    # combine = 'both' without bulk samples
    expect_error(
      gridSearch(
        object = DDLS, combine = "both", params = params, 
      regexp = "If 'combine = both' is selected, 'bulk.simul' and at least one single cell slot must be provided"
    ))
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
        expect_error(
            gridSearch(DDLS, params, prop.test = 2),
            regexp = "'prop.test' must be a number between 0 and 1"
        )
        expect_error(
            gridSearch(DDLS, params, prop.val = -2),
            regexp = "'prop.val' must be a number between 0 and 1"
        )
    }
)

test_that(
    "Test behaviour",
    {
        probMatrixValid <- data.frame(
        Cell_Type = paste0("CellType", seq(4)),
        from = c(1, 1, 1, 30),
        to = c(15, 15, 50, 70)
        )
        DDLS <- generateBulkCellMatrix(
            object = DDLS,
            cell.ID.column = "Cell_ID",
            cell.type.column = "Cell_Type",
            prob.design = probMatrixValid,
            num.bulk.samples = 100,
            verbose = FALSE
        )
        DDLS <- simBulkProfiles(DDLS, verbose = FALSE)

        # On the fly option but no on the fly parameters
        expect_error(
            gridSearch(object = DDLS, params = params, on.the.fly = TRUE),
            regexp = "'on.the.fly' option selected but no given parameters for on the fly bulk generation functions"
        )
        expect_message(
            gridSearch(object = DDLS, params = params, view.metrics.plot = FALSE),
            regexp = "Starting grid search with 2 different models"
        )
        expect_warning(
            gridSearch(object = DDLS, params = params, subset = 1000, view.metrics.plot = FALSE),
            regexp = "'subset' parameter is greater than then number of pseudobulks. All pseudobulks are being used for train"
        )
    }
)