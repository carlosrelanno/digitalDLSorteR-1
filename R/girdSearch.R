gridSearch <- function(
  object,
  params,
  subset,
  verbose = TRUE
) {
  # generar dataframe de combinaciones
  combinations <- expand.grid(params)
}




.targetForDNNSubset <- function(
  object, 
  combine,
  type.data,
  fly,
  shuffle,
  sample.subset,
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
  # shuffle only if train on the fly
  if (shuffle && !fly) {
    return(probs.matrix[sample(nrow(probs.matrix)), ])
  } else {
    return(probs.matrix)
  }
}


