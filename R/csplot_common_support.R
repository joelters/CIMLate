#' Internal helper to plot common support
#'
#' @param ps Numeric propensity score vector
#' @param D Binary treatment indicator
#'
#' @return A recorded base plot object
#' @keywords internal
csplot_common_support <- function(ps, D) {
  c1 <- grDevices::rgb(173, 216, 230, max = 255, alpha = 180)
  c2 <- grDevices::rgb(255, 192, 203, max = 255, alpha = 180)

  graphics::par(mar = c(5, 5, 5, 5) + 0.3)
  graphics::hist(ps[D == 1], freq = TRUE, col = c1, axes = FALSE,
                 xlab = "", ylab = "", main = "")
  graphics::axis(side = 1)
  graphics::axis(side = 4, labels = FALSE)
  graphics::mtext(side = 4, text = "D=1", line = 2.5, col = "blue")

  graphics::par(new = TRUE)
  graphics::hist(ps[D == 0], freq = TRUE, axes = FALSE, col = c2,
                 xlab = "", ylab = "", main = "Common Support")
  graphics::axis(side = 2)
  graphics::mtext(side = 2, text = "D=0", line = 2.5, col = "pink")

  grDevices::recordPlot()
}
