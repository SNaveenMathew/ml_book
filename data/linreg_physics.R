# Initialization
set.seed(1)
n_train <- 100
x1 <- 1:n_train
error_sd <- 10
res <- rnorm(n = length(x1), mean = 0, sd = error_sd)
w_0 <- 0
w_1 <- 2
y <- w_0 + w_1 * x1 + res
df <- data.frame(x1 = x1, y = y)

# Fitting the model
model <- lm(formula = y ~ x1, data = df)
df$predicted <- predict(model)
df$residuals <- residuals(model)

# Plotting
library(ggplot2)
library(latex2exp)
library(gganimate)

bar_y <- mean(y)
bar_x1 <- mean(x1)
theta_inst <- 0
omega_inst <- 0
w_1 <- tan(theta_inst)
w_0 <- bar_y - w_1 * bar_x1
control_inertia <- 12
dt <- 0.001
I_total <- (max(df$x1) - min(df$x1))^3/control_inertia
plots <- list()
rows <- nrow(df)
n_iterations <- 10000
df_time <- data.frame(matrix(0, nrow = n_iterations * rows, ncol = 6))
colnames(df_time) <- c("x1", "y", "predicted", "residuals", "pred_y_inst", "time")

for(i in 1:n_iterations) {
  df$pred_y_inst <- w_0 + w_1 * x1
  df$residuals <- df$y - df$pred_y_inst
  df$time <- i
  df_time[(rows * (i - 1) + 1):(rows * i), ] <- df
  force <- round(sum(df$residuals), 4)
  torque <- sum((df$x1 - bar_x1) * df$residuals)
  if(i %% 100 == 0)
    print(c(i = i, force = force))
  
  alpha <- torque/I_total
  omega_inst <- omega_inst + alpha * dt
  # Very rough model, assuming constant alpha throughout
  theta_inst <- theta_inst + omega_inst * dt
  w_1 <- tan(theta_inst)
  w_0 <- bar_y - w_1 * bar_x1
}

ggplot(df_time, aes(x = x1, y = y)) + transition_time(time) +
  geom_segment(aes(xend = x1, yend = pred_y_inst), alpha = .2) +
  labs(title = 'Iteration: {frame_time}', x = TeX("$x_1$"), y = 'y') +
  geom_point() + ggplot2::xlim(range(x1))
anim_save("physics_anim.gif")
