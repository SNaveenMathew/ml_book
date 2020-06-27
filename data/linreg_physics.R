# Initialization
set.seed(1)
n_train <- 100
x1 <- 1:n_train
error_sd <- 10
res <- rnorm(n = length(x1), mean = 0, sd = error_sd)
W_0 <- 0
W_1 <- 2
y <- W_0 + W_1 * x1 + res
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
alpha_prev <- 0 # Clearly wrong because this should be high, but ok!
w_1 <- tan(theta_inst_prev)
w_0 <- bar_y - w_1 * bar_x1
control_inertia <- 0.1
dt <- 0.01
I_total <- (max(df$x1) - min(df$x1))^3/control_inertia
plots <- list()
rows <- nrow(df)
n_iterations <- 3000
df_time <- data.frame(matrix(0, nrow = rows, ncol = 4))
colnames(df_time) <- c("x1", "y", "predicted", "residuals")
df_time$x1 <- x1
df_time$y <- y
w_0s <- w_1s <- forces <- torques <- alphas <- omegas <- thetas <- rep(0, n_iterations)

for(i in 1:n_iterations) {
  pred_y_inst <- w_0 + w_1 * x1
  residuals <- y - pred_y_inst
  force <- round(sum(residuals), 4)
  torque <- sum((x1 - bar_x1) * residuals)
  forces[i] <- force
  torques[i] <- torque
    
  if(i %% 1000 == 0)
    print(c(i = i, force = force))
  
  alpha_next <- torque/I_total
  alphas[i] <- alpha_next
  # Using mid-point estimate for $\alpha$ within interval dt
  omega_inst_next <- omega_inst_prev + 0.5 * (alpha_prev + alpha_next) * dt
  omegas[i] <- omega_inst_next
  # Using mid-point estimate for $\Omega$ within interval dt
  theta_inst_next <- theta_inst_prev + 0.5 * (omega_inst_next + omega_inst_prev) * dt
  w_1 <- tan(theta_inst_next)
  w_0 <- bar_y - w_1 * bar_x1
  w_0s[i] <- w_0
  w_1s[i] <- w_1
  omega_inst_prev <- omega_inst_next
  theta_inst_prev <- theta_inst_next
  alpha_prev <- alpha_next
}
df <- data.frame(time = 1:n_iterations, force = forces, torque = torques,
                 alpha = alphas, omega = omegas, theta = thetas,
                 w_0 = w_0s, w_1 = w_1s)
df_time <- merge(df_time, df)
df_time$predicted <- df_time$w_0 + df_time$w_1 * df_time$x1

p <- ggplot(df_time, aes(x = x1, y = y)) + transition_time(time) +
  geom_segment(aes(xend = x1, yend = predicted), alpha = .2) +
  labs(title = 'Iteration: {frame_time}', x = TeX("$x_1$"), y = 'y') +
  geom_point() + ggplot2::xlim(range(x1))
anim_save(filename = "physics_anim.gif", animation = p)
