% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gridSearch.R
\name{gridParamDist}
\alias{gridParamDist}
\title{Histogram of the distribution of a given parameter}
\usage{
gridParamDist(object, param, theme = NULL, ...)
}
\arguments{
\item{object}{\code{\linkS4class{DigitalDLSorter}} object with
\code{grid.search} slot.}

\item{param}{The parameter to plot. The available parameters can be known
using \code{params()}}

\item{theme}{\pkg{ggplot2} theme.}

\item{...}{}
}
\description{
Display the distribution of a given parameter between all generated models.
}
