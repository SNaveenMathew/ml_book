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

# This simulation was run on a server with high RAM
# Ideal: Reduce n_iterations by a factor of 10 and increase dt by a factor of 10
bar_y <- mean(y)
bar_x1 <- mean(x1)
theta_inst_prev <- 0
omega_inst_prev <- 0
w_1 <- tan(theta_inst_prev)
w_0 <- bar_y - w_1 * bar_x1
control_inertia <- 12
dt <- 0.0001
I_total <- (max(df$x1) - min(df$x1))^3/control_inertia
plots <- list()
rows <- nrow(df)
n_iterations <- 50000
df_time <- data.frame(matrix(0, nrow = n_iterations * rows, ncol = 6))
colnames(df_time) <- c("x1", "y", "predicted", "residuals", "pred_y_inst", "time")

for(i in 1:n_iterations) {
  df$pred_y_inst <- w_0 + w_1 * df$x1
  df$residuals <- df$y - df$pred_y_inst
  df$time <- i
  df_time[(rows * (i - 1) + 1):(rows * i), ] <- df
  force <- round(sum(df$residuals), 4)
  torque <- sum((df$x1 - bar_x1) * df$residuals)
  if(i %% 1000 == 0)
    print(c(i = i, force = force))
  
  alpha_next <- torque/I_total
  if(i == 1) {
    alpha_prev <- alpha_next
  }
  # Using mid-point estimate for $\alpha$ within interval dt
  omega_inst_next <- omega_inst_prev + 0.5 * (alpha_prev + alpha_next) * dt
  # Using mid-point estimate for $\Omega$ within interval dt
  theta_inst_next <- theta_inst_prev + 0.5 * (omega_inst_next + omega_inst_prev) * dt
  w_1 <- tan(theta_inst_next)
  w_0 <- bar_y - w_1 * bar_x1
  omega_inst_prev <- omega_inst_next
  theta_inst_prev <- theta_inst_next
  alpha_prev <- alpha_next
}

p <- ggplot(df_time, aes(x = x1, y = y)) + transition_time(time) +
  geom_segment(aes(xend = x1, yend = pred_y_inst), alpha = .2) +
  labs(title = 'Iteration: {frame_time}', x = TeX("$x_1$"), y = 'y') +
  geom_point() + ggplot2::xlim(range(x1))
anim_save(filename = "physics_anim.gif", animation = p)
