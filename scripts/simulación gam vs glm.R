#Comparación entre regresión logística y modelos aditivos generalizados 
# en datos binarios desbalanceados con predictores no lineales. (simulaciones)


#OBSERVACIONES PARA CORRER CÓDIGOS DE LOS 3 LOOPS:

   #1. Las librerias, la semilla, funciones de calibración y parámetros fijos son los mismos
    #  para los tres casos: débil(polinomio de 2°), suave(Componente sinusoidal), fuerte
    #  (Polinomio de 3°). 

   #2. NO están diseñados para correr los tres loops al mismo tiempo, se debe eliminar
   #   la información de uno para correr el otro loop ya que tienen los mismo nombres.

   #3. Para cada tanda de loop se incluye el aspecto LINEAL para confirmar la información
   #   predicha.



library(mgcv)
library(pROC)
library(mvtnorm)
library(dplyr)

set.seed(7)

# ==============================
# FUNCIONES DE CALIBRACIÓN LINEAR Y NO LINEAL
# ==============================

#El código crea una función donde calibra mediante la fórmula del plan de simulación, 
#el valor que debe llevar el beta cero en cada combinación de factores.

# Esto se discrimina entre interceptos del componente lineal y no lineal.

calibrar_intercepto_lineal <- function(pi_objetivo, eta) {
  f <- function(beta0) {
    mean(1 / (1 + exp(-(beta0 + eta)))) - pi_objetivo
  }
  uniroot(f, c(-20,20), extendInt = "yes")$root
}


calibrar_intercepto_nolineal <- function(pi_objetivo, eta) {
  f <- function(beta0) {
    mean(1 / (1 + exp(-(beta0 + eta)))) - pi_objetivo
  }
  uniroot(f, c(-20,20), extendInt = "yes")$root
}

metricas_cm <- function(y_real, y_pred){
  
  cm <- table(factor(y_real, levels=c(0,1)),
              factor(y_pred, levels=c(0,1)))
  
  TN <- cm[1,1]
  FP <- cm[1,2]
  FN <- cm[2,1]
  TP <- cm[2,2]
  
  recall <- ifelse((TP+FN)>0, TP/(TP+FN), NA)
  precision <- ifelse((TP+FP)>0, TP/(TP+FP), NA)
  f1 <- ifelse((precision+recall)>0, 
               2*(precision*recall)/(precision+recall), NA)
  specificity <- ifelse((TN+FP)>0, TN/(TN+FP), NA)
  bal_acc <- (recall + specificity)/2
  
  return(c(recall, precision, f1, bal_acc))
}
# ==============================
# PARÁMETROS FIJOS
# ==============================

beta1 <- 0.05
beta2 <- 0.12
efecto_sede <- c(A=0, B=0.4, C=-0.3)

resultados <- data.frame()

# ==============================
# LOOP PARA EL CASO DE NO LINEALIDAD DÉBIL
# ==============================

for(n in c(50,100,300,600,1000)){
  
  for(rho in c(0.3,0.8)){
    
    for(pi_objetivo in c(0.5,0.15,0.05)){
      
      for(rep in 1:100){
        
        set.seed(rep)
        
        # ==============================
        # GENERAR DATOS (NORMAL MULTIVARIADA)
        # ==============================
        
        mu <- c(68, 26)
        
        Sigma <- matrix(c(12^2, rho*12*6,
                          rho*12*6, 6^2), 2, 2)
        
        datos_cov <- rmvnorm(n, mean = mu, sigma = Sigma)
        
        Edad <- round(pmin(pmax(datos_cov[,1],19),98))
        IMC  <- round(pmin(pmax(datos_cov[,2],15),53),1)
        
        Sede <- sample(c("A","B","C"), n, replace=TRUE,
                       prob=c(0.4,0.35,0.25))
        
        # ==============================
        # MODELO LINEAL
        # ==============================
        
        eta_lineal <- beta1*Edad + beta2*IMC + efecto_sede[Sede]
        beta0 <- calibrar_intercepto_lineal(pi_objetivo, eta_lineal)
        
        p <- 1/(1+exp(-(beta0 + eta_lineal)))
        Y_lineal <- rbinom(n,1,p)
        
        datos_lineal <- data.frame(Edad,IMC,Sede=factor(Sede),Y=Y_lineal)
        
        # ==============================
        # MODELO NO LINEAL
        # ==============================
        
        Edad_c <- Edad - mean(Edad)
        IMC_c  <- IMC - mean(IMC)
        
        f1 <- 0.03 * Edad_c - 0.0012 * (Edad_c^2)
        f2 <- 0.08 * IMC_c  - 0.015  * (IMC_c^2)
        
        eta_nl <- f1 + f2 + efecto_sede[Sede]
        beta0_nl <- calibrar_intercepto_nolineal(pi_objetivo, eta_nl)
        
        p_nl <- 1/(1+exp(-(beta0_nl + eta_nl)))
        Y_nolineal <- rbinom(n,1,p_nl)
        
        datos_nolineal <- data.frame(Edad,IMC,Sede=factor(Sede),Y=Y_nolineal)
        
        # ==============================
        # TRAIN / TEST
        # ==============================
        
        idx <- sample(1:n,0.7*n)
        train_l <- datos_lineal[idx,]
        test_l  <- datos_lineal[-idx,]
        
        idx <- sample(1:n,0.7*n)
        train_nl <- datos_nolineal[idx,]
        test_nl  <- datos_nolineal[-idx,]
        
        # ==============================
        # MODELOS
        # ==============================
        
        glm1 <- glm(Y~Edad+IMC+Sede, data=train_l, family=binomial()) #M. logístico(Lineal)
        glm2 <- glm(Y~Edad+IMC+Sede, data=train_nl, family=binomial()) #M. logístico(no lineal)
        
        gam1 <- gam(Y~s(Edad)+s(IMC)+Sede, data=train_l, family=binomial())  #M. GAM(lineal)
        gam2 <- gam(Y~s(Edad)+s(IMC)+Sede, data=train_nl, family=binomial()) #M. GAM(no lineal)
        
        # ==============================
        # PREDICCIONES
        # ==============================
        
        pred_glm1 <- predict(glm1,test_l,type="response")
        pred_gam1 <- predict(gam1,test_l,type="response")
        pred_glm2 <- predict(glm2,test_nl,type="response")
        pred_gam2 <- predict(gam2,test_nl,type="response")
        
        # ==============================
        # AUC
        # ==============================
        
        auc_glm1 <- if(length(unique(test_l$Y))==2) auc(roc(test_l$Y,pred_glm1)) else NA
        auc_gam1 <- if(length(unique(test_l$Y))==2) auc(roc(test_l$Y,pred_gam1)) else NA
        auc_glm2 <- if(length(unique(test_nl$Y))==2) auc(roc(test_nl$Y,pred_glm2)) else NA
        auc_gam2 <- if(length(unique(test_nl$Y))==2) auc(roc(test_nl$Y,pred_gam2)) else NA
        
        # ==============================
        # ACC
        # ==============================
        
        acc_glm1 <- mean((pred_glm1>0.5)==test_l$Y)
        acc_gam1 <- mean((pred_gam1>0.5)==test_l$Y)
        acc_glm2 <- mean((pred_glm2>0.5)==test_nl$Y)
        acc_gam2 <- mean((pred_gam2>0.5)==test_nl$Y)
        
        # Clasificación
        class_glm1 <- ifelse(pred_glm1 > 0.5, 1, 0)
        class_gam1 <- ifelse(pred_gam1 > 0.5, 1, 0)
        class_glm2 <- ifelse(pred_glm2 > 0.5, 1, 0)
        class_gam2 <- ifelse(pred_gam2 > 0.5, 1, 0)
        
        # Métricas desde matriz de confusión
        m_glm1 <- metricas_cm(test_l$Y, class_glm1)
        m_gam1 <- metricas_cm(test_l$Y, class_gam1)
        m_glm2 <- metricas_cm(test_nl$Y, class_glm2)
        m_gam2 <- metricas_cm(test_nl$Y, class_gam2)
        
        # ==============================
        # AIC Y RESULTADOS FINALES
        # ==============================
        
        resultados <- rbind(resultados, data.frame(
          n=n,
          correlacion=rho,
          desbalance=pi_objetivo,
          replica=rep,
          
          # AUC
          AUC_GLM1=auc_glm1,
          AUC_GAM1=auc_gam1,
          AUC_GLM2=auc_glm2,
          AUC_GAM2=auc_gam2,
          
          # ACC
          ACC_GLM1=acc_glm1,
          ACC_GAM1=acc_gam1,
          ACC_GLM2=acc_glm2,
          ACC_GAM2=acc_gam2,
          
          # NUEVAS MÉTRICAS
          Recall_GLM1=m_glm1[1],
          Recall_GAM1=m_gam1[1],
          Recall_GLM2=m_glm2[1],
          Recall_GAM2=m_gam2[1],
          
          Precision_GLM1=m_glm1[2],
          Precision_GAM1=m_gam1[2],
          Precision_GLM2=m_glm2[2],
          Precision_GAM2=m_gam2[2],
          
          F1_GLM1=m_glm1[3],
          F1_GAM1=m_gam1[3],
          F1_GLM2=m_glm2[3],
          F1_GAM2=m_gam2[3],
          
          BalAcc_GLM1=m_glm1[4],
          BalAcc_GAM1=m_gam1[4],
          BalAcc_GLM2=m_glm2[4],
          BalAcc_GAM2=m_gam2[4],
          
          # AIC
          AIC_GLM1=AIC(glm1),
          AIC_GAM1=AIC(gam1),
          AIC_GLM2=AIC(glm2),
          AIC_GAM2=AIC(gam2)
        ))
        
      }
    }
  }
}

# ==============================
# TABLA FINAL
# ==============================

tabla_final <- resultados %>%
  group_by(n, correlacion, desbalance) %>%
  summarise(across(starts_with(c("AUC","ACC","AIC",
                                 "Recall","Precision",
                                 "F1","BalAcc")),
                   mean, na.rm=TRUE))

tabla_final



#* Se debe eliminar la información anterior*

# ==============================
# LOOP PARA EL CASO DE NO LINEALIDAD SUAVE
# ==============================
for(n in c(50,100,300,600,1000)){
  
  for(rho in c(0.3,0.8)){
    
    for(pi_objetivo in c(0.5,0.15,0.05)){
      
      for(rep in 1:100){
        
        set.seed(rep)
        
        # ==============================
        # GENERAR DATOS (NORMAL MULTIVARIADA)
        # ==============================
        
        mu <- c(68, 26)
        
        Sigma <- matrix(c(12^2, rho*12*6,
                          rho*12*6, 6^2), 2, 2)
        
        datos_cov <- rmvnorm(n, mean = mu, sigma = Sigma)
        
        Edad <- round(pmin(pmax(datos_cov[,1],19),98))
        IMC  <- round(pmin(pmax(datos_cov[,2],15),53),1)
        
        Sede <- sample(c("A","B","C"), n, replace=TRUE,
                       prob=c(0.4,0.35,0.25))
        
        # ==============================
        # MODELO LINEAL
        # ==============================
        
        eta_lineal <- beta1*Edad + beta2*IMC + efecto_sede[Sede]
        beta0 <- calibrar_intercepto_lineal(pi_objetivo, eta_lineal)
        
        p <- 1/(1+exp(-(beta0 + eta_lineal)))
        Y_lineal <- rbinom(n,1,p)
        
        datos_lineal <- data.frame(Edad,IMC,Sede=factor(Sede),Y=Y_lineal)
        
        # ==============================
        # MODELO NO LINEAL
        # ==============================
        
        Edad_c <- Edad - mean(Edad)
        IMC_c  <- IMC - mean(IMC)
        
        f1 <- sin(Edad_c / 10)
        f2 <- cos(IMC_c / 5)
        
        eta_nl <- f1 + f2 + efecto_sede[Sede]
        beta0_nl <- calibrar_intercepto_nolineal(pi_objetivo, eta_nl)
        
        p_nl <- 1/(1+exp(-(beta0_nl + eta_nl)))
        Y_nolineal <- rbinom(n,1,p_nl)
        
        datos_nolineal <- data.frame(Edad,IMC,Sede=factor(Sede),Y=Y_nolineal)
        
        # ==============================
        # TRAIN / TEST
        # ==============================
        
        idx <- sample(1:n,0.7*n)
        train_l <- datos_lineal[idx,]
        test_l  <- datos_lineal[-idx,]
        
        idx <- sample(1:n,0.7*n)
        train_nl <- datos_nolineal[idx,]
        test_nl  <- datos_nolineal[-idx,]
        
        # ==============================
        # MODELOS
        # ==============================
        
        glm1 <- glm(Y~Edad+IMC+Sede, data=train_l, family=binomial()) #M. logístico(Lineal)
        glm2 <- glm(Y~Edad+IMC+Sede, data=train_nl, family=binomial()) #M. logístico(no lineal)
        
        gam1 <- gam(Y~s(Edad)+s(IMC)+Sede, data=train_l, family=binomial())  #M. GAM(lineal)
        gam2 <- gam(Y~s(Edad)+s(IMC)+Sede, data=train_nl, family=binomial()) #M. GAM(no lineal)
        
        # ==============================
        # PREDICCIONES
        # ==============================
        
        pred_glm1 <- predict(glm1,test_l,type="response")
        pred_gam1 <- predict(gam1,test_l,type="response")
        pred_glm2 <- predict(glm2,test_nl,type="response")
        pred_gam2 <- predict(gam2,test_nl,type="response")
        
        # ==============================
        # AUC
        # ==============================
        
        auc_glm1 <- if(length(unique(test_l$Y))==2) auc(roc(test_l$Y,pred_glm1)) else NA
        auc_gam1 <- if(length(unique(test_l$Y))==2) auc(roc(test_l$Y,pred_gam1)) else NA
        auc_glm2 <- if(length(unique(test_nl$Y))==2) auc(roc(test_nl$Y,pred_glm2)) else NA
        auc_gam2 <- if(length(unique(test_nl$Y))==2) auc(roc(test_nl$Y,pred_gam2)) else NA
        
        # ==============================
        # ACC
        # ==============================
        
        acc_glm1 <- mean((pred_glm1>0.5)==test_l$Y)
        acc_gam1 <- mean((pred_gam1>0.5)==test_l$Y)
        acc_glm2 <- mean((pred_glm2>0.5)==test_nl$Y)
        acc_gam2 <- mean((pred_gam2>0.5)==test_nl$Y)
        
        # Clasificación
        class_glm1 <- ifelse(pred_glm1 > 0.5, 1, 0)
        class_gam1 <- ifelse(pred_gam1 > 0.5, 1, 0)
        class_glm2 <- ifelse(pred_glm2 > 0.5, 1, 0)
        class_gam2 <- ifelse(pred_gam2 > 0.5, 1, 0)
        
        # Métricas desde matriz de confusión
        m_glm1 <- metricas_cm(test_l$Y, class_glm1)
        m_gam1 <- metricas_cm(test_l$Y, class_gam1)
        m_glm2 <- metricas_cm(test_nl$Y, class_glm2)
        m_gam2 <- metricas_cm(test_nl$Y, class_gam2)
        
        # ==============================
        # AIC Y RESULTADOS FINALES
        # ==============================
        
        resultados <- rbind(resultados, data.frame(
          n=n,
          correlacion=rho,
          desbalance=pi_objetivo,
          replica=rep,
          
          # AUC
          AUC_GLM1=auc_glm1,
          AUC_GAM1=auc_gam1,
          AUC_GLM2=auc_glm2,
          AUC_GAM2=auc_gam2,
          
          # ACC
          ACC_GLM1=acc_glm1,
          ACC_GAM1=acc_gam1,
          ACC_GLM2=acc_glm2,
          ACC_GAM2=acc_gam2,
          
          # NUEVAS MÉTRICAS
          Recall_GLM1=m_glm1[1],
          Recall_GAM1=m_gam1[1],
          Recall_GLM2=m_glm2[1],
          Recall_GAM2=m_gam2[1],
          
          Precision_GLM1=m_glm1[2],
          Precision_GAM1=m_gam1[2],
          Precision_GLM2=m_glm2[2],
          Precision_GAM2=m_gam2[2],
          
          F1_GLM1=m_glm1[3],
          F1_GAM1=m_gam1[3],
          F1_GLM2=m_glm2[3],
          F1_GAM2=m_gam2[3],
          
          BalAcc_GLM1=m_glm1[4],
          BalAcc_GAM1=m_gam1[4],
          BalAcc_GLM2=m_glm2[4],
          BalAcc_GAM2=m_gam2[4],
          
          # AIC
          AIC_GLM1=AIC(glm1),
          AIC_GAM1=AIC(gam1),
          AIC_GLM2=AIC(glm2),
          AIC_GAM2=AIC(gam2)
        ))
        
      }
    }
  }
}

# ==============================
# TABLA FINAL
# ==============================

tabla_final <- resultados %>%
  group_by(n, correlacion, desbalance) %>%
  summarise(across(starts_with(c("AUC","ACC","AIC",
                                 "Recall","Precision",
                                 "F1","BalAcc")),
                   mean, na.rm=TRUE))

tabla_final

#* Se debe eliminar la información de los loops anteriores para correr este loop*

# ==============================
# LOOP PARA EL CASO DE NO LINEALIDAD FUERTE
# ==============================

for(n in c(50,100,300,600,1000)){
  
  for(rho in c(0.3,0.8)){
    
    for(pi_objetivo in c(0.5,0.15,0.05)){
      
      for(rep in 1:100){
        
        set.seed(rep)
        
        # ==============================
        # GENERAR DATOS (NORMAL MULTIVARIADA)
        # ==============================
        
        mu <- c(68, 26)
        
        Sigma <- matrix(c(12^2, rho*12*6,
                          rho*12*6, 6^2), 2, 2)
        
        datos_cov <- rmvnorm(n, mean = mu, sigma = Sigma)
        
        Edad <- round(pmin(pmax(datos_cov[,1],19),98))
        IMC  <- round(pmin(pmax(datos_cov[,2],15),53),1)
        
        Sede <- sample(c("A","B","C"), n, replace=TRUE,
                       prob=c(0.4,0.35,0.25))
        
        # ==============================
        # MODELO LINEAL
        # ==============================
        
        eta_lineal <- beta1*Edad + beta2*IMC + efecto_sede[Sede]
        beta0 <- calibrar_intercepto_lineal(pi_objetivo, eta_lineal)
        
        p <- 1/(1+exp(-(beta0 + eta_lineal)))
        Y_lineal <- rbinom(n,1,p)
        
        datos_lineal <- data.frame(Edad,IMC,Sede=factor(Sede),Y=Y_lineal)
        
        # ==============================
        # MODELO NO LINEAL
        # ==============================
        
        Edad_c <- Edad - mean(Edad)
        IMC_c  <- IMC - mean(IMC)
        
        f1 <- 0.0005 * Edad_c^3 - 0.01 * Edad_c^2
        f2 <- 0.0008 * IMC_c^3  - 0.02 * IMC_c^2
        
        eta_nl <- f1 + f2 + efecto_sede[Sede]
        beta0_nl <- calibrar_intercepto_nolineal(pi_objetivo, eta_nl)
        
        p_nl <- 1/(1+exp(-(beta0_nl + eta_nl)))
        Y_nolineal <- rbinom(n,1,p_nl)
        
        datos_nolineal <- data.frame(Edad,IMC,Sede=factor(Sede),Y=Y_nolineal)
        
        # ==============================
        # TRAIN / TEST
        # ==============================
        
        idx <- sample(1:n,0.7*n)
        train_l <- datos_lineal[idx,]
        test_l  <- datos_lineal[-idx,]
        
        idx <- sample(1:n,0.7*n)
        train_nl <- datos_nolineal[idx,]
        test_nl  <- datos_nolineal[-idx,]
        
        # ==============================
        # MODELOS
        # ==============================
        
        glm1 <- glm(Y~Edad+IMC+Sede, data=train_l, family=binomial()) #M. logístico(Lineal)
        glm2 <- glm(Y~Edad+IMC+Sede, data=train_nl, family=binomial()) #M. logístico(no lineal)
        
        gam1 <- gam(Y~s(Edad)+s(IMC)+Sede, data=train_l, family=binomial())  #M. GAM(lineal)
        gam2 <- gam(Y~s(Edad)+s(IMC)+Sede, data=train_nl, family=binomial()) #M. GAM(no lineal)
        
        # ==============================
        # PREDICCIONES
        # ==============================
        
        pred_glm1 <- predict(glm1,test_l,type="response")
        pred_gam1 <- predict(gam1,test_l,type="response")
        pred_glm2 <- predict(glm2,test_nl,type="response")
        pred_gam2 <- predict(gam2,test_nl,type="response")
        
        # ==============================
        # AUC
        # ==============================
        
        auc_glm1 <- if(length(unique(test_l$Y))==2) auc(roc(test_l$Y,pred_glm1)) else NA
        auc_gam1 <- if(length(unique(test_l$Y))==2) auc(roc(test_l$Y,pred_gam1)) else NA
        auc_glm2 <- if(length(unique(test_nl$Y))==2) auc(roc(test_nl$Y,pred_glm2)) else NA
        auc_gam2 <- if(length(unique(test_nl$Y))==2) auc(roc(test_nl$Y,pred_gam2)) else NA
        
        # ==============================
        # ACC
        # ==============================
        
        acc_glm1 <- mean((pred_glm1>0.5)==test_l$Y)
        acc_gam1 <- mean((pred_gam1>0.5)==test_l$Y)
        acc_glm2 <- mean((pred_glm2>0.5)==test_nl$Y)
        acc_gam2 <- mean((pred_gam2>0.5)==test_nl$Y)
        
        # Clasificación
        class_glm1 <- ifelse(pred_glm1 > 0.5, 1, 0)
        class_gam1 <- ifelse(pred_gam1 > 0.5, 1, 0)
        class_glm2 <- ifelse(pred_glm2 > 0.5, 1, 0)
        class_gam2 <- ifelse(pred_gam2 > 0.5, 1, 0)
        
        # Métricas desde matriz de confusión
        m_glm1 <- metricas_cm(test_l$Y, class_glm1)
        m_gam1 <- metricas_cm(test_l$Y, class_gam1)
        m_glm2 <- metricas_cm(test_nl$Y, class_glm2)
        m_gam2 <- metricas_cm(test_nl$Y, class_gam2)
        
        # ==============================
        # AIC Y RESULTADOS FINALES
        # ==============================
        
        resultados <- rbind(resultados, data.frame(
          n=n,
          correlacion=rho,
          desbalance=pi_objetivo,
          replica=rep,
          
          # AUC
          AUC_GLM1=auc_glm1,
          AUC_GAM1=auc_gam1,
          AUC_GLM2=auc_glm2,
          AUC_GAM2=auc_gam2,
          
          # ACC
          ACC_GLM1=acc_glm1,
          ACC_GAM1=acc_gam1,
          ACC_GLM2=acc_glm2,
          ACC_GAM2=acc_gam2,
          
          # NUEVAS MÉTRICAS
          Recall_GLM1=m_glm1[1],
          Recall_GAM1=m_gam1[1],
          Recall_GLM2=m_glm2[1],
          Recall_GAM2=m_gam2[1],
          
          Precision_GLM1=m_glm1[2],
          Precision_GAM1=m_gam1[2],
          Precision_GLM2=m_glm2[2],
          Precision_GAM2=m_gam2[2],
          
          F1_GLM1=m_glm1[3],
          F1_GAM1=m_gam1[3],
          F1_GLM2=m_glm2[3],
          F1_GAM2=m_gam2[3],
          
          BalAcc_GLM1=m_glm1[4],
          BalAcc_GAM1=m_gam1[4],
          BalAcc_GLM2=m_glm2[4],
          BalAcc_GAM2=m_gam2[4],
          
          # AIC
          AIC_GLM1=AIC(glm1),
          AIC_GAM1=AIC(gam1),
          AIC_GLM2=AIC(glm2),
          AIC_GAM2=AIC(gam2)
        ))
        
      }
    }
  }
}

# ==============================
# TABLA FINAL
# ==============================

tabla_final <- resultados %>%
  group_by(n, correlacion, desbalance) %>%
  summarise(across(starts_with(c("AUC","ACC","AIC",
                                 "Recall","Precision",
                                 "F1","BalAcc")),
                   mean, na.rm=TRUE))

tabla_final



# ==============================
# RESULTADOS EN DATOS REALES
# ==============================


library(readxl)
library(caret)
library(pROC)
library(mgcv)  

datos <- read_excel("C:/Users/TATIANA/Downloads/Enfermedades111.xltx")

# Selección de variables
datos <- datos[, c("DISCAPACIDAD", "EDAD", "IMC", "SEDE")]

# Conversión
datos$DISCAPACIDAD <- as.factor(datos$DISCAPACIDAD)
datos$SEDE <- as.factor(datos$SEDE)
datos$IMC<- as.numeric(datos$IMC)

# Asegurar que sea binaria (0/1)
datos$DISCAPACIDAD <- ifelse(datos$DISCAPACIDAD == "No", 0, 1)
datos$DISCAPACIDAD <- as.factor(datos$DISCAPACIDAD)


set.seed(123)

k <- 5
folds <- createFolds(datos$DISCAPACIDAD, k = k)

auc_glm <- numeric(k)
auc_gam <- numeric(k)

for(i in 1:k){
  
  test_idx <- folds[[i]]
  
  train <- datos[-test_idx, ]
  test  <- datos[test_idx, ]
  
  # -------------------
  # 🔹 MODELO LOGÍSTICO
  # -------------------
  modelo_glm <- glm(DISCAPACIDAD ~ EDAD + IMC + SEDE,
                    data = train,
                    family = binomial)
  
  pred_glm <- predict(modelo_glm, newdata = test, type = "response")
  
  # -------------------
  # 🔹 MODELO GAM
  # -------------------
  modelo_gam <- gam(DISCAPACIDAD ~ s(EDAD) + s(IMC) + SEDE,
                    data = train,
                    family = binomial)
  
  pred_gam <- predict(modelo_gam, newdata = test, type = "response")
  
  # -------------------
  # 🔹 AUC
  # -------------------
  if(length(unique(test$DISCAPACIDAD)) == 2){
    
    auc_glm[i] <- auc(roc(test$DISCAPACIDAD, pred_glm))
    auc_gam[i] <- auc(roc(test$DISCAPACIDAD, pred_gam))
    
  } else {
    auc_glm[i] <- NA
    auc_gam[i] <- NA
  }
}

mean(auc_glm, na.rm = TRUE)
mean(auc_gam, na.rm = TRUE)

auc_glm
auc_gam

t.test(auc_glm, auc_gam, paired = TRUE)

mean(auc_gam) > mean(auc_glm)

summary(modelo_glm)
summary(modelo_gam)

# Odds Ratios
exp(coef(modelo_glm))

# Intervalos de confianza
exp(confint(modelo_glm))

# ==============================
# AUC
# ==============================

roc_glm <- roc(test$DISCAPACIDAD, pred_glm)
roc_gam <- roc(test$DISCAPACIDAD, pred_gam)

auc_glm <- as.numeric(auc(roc_glm))
auc_gam <- as.numeric(auc(roc_gam))

# ==============================
# ACCURACY
# ==============================

pred_class_glm <- ifelse(pred_glm > 0.5, 1, 0)
pred_class_gam <- ifelse(pred_gam > 0.5, 1, 0)

acc_glm <- mean(pred_class_glm == test$DISCAPACIDAD)
acc_gam <- mean(pred_class_gam == test$DISCAPACIDAD)

# ==============================
# AIC (sobre datos de entrenamiento)
# ==============================

AIC_glm <- AIC(modelo_glm)
AIC_gam <- AIC(modelo_gam)

# ==============================
# TABLA DE RESULTADOS
# ==============================

resultados <- data.frame(
  Modelo   = c("Logístico (GLM)", "GAM"),
  AUC      = c(auc_glm, auc_gam),
  Accuracy = c(acc_glm, acc_gam),
  AIC      = c(AIC_glm, AIC_gam)
)

print(resultados)

# 2. Extraer coeficientes, errores estándar y p-valores
resumen <- summary(modelo_glm)$coefficients

# 3. Calcular los Odds Ratios (exponencial de los coeficientes)
odds_ratios <- exp(coef(modelo_glm))

# 4. Calcular los Intervalos de Confianza al 95% para los Odds Ratios
intervalos_confianza <- exp(confint(modelo_glm))

# 5. Consolidar todo en una tabla limpia
tabla_or <- data.frame(
  Variable = rownames(resumen),
  Estimado_Beta = resumen[, 1],
  Odds_Ratio = odds_ratios,
  IC_Inferior_95 = intervalos_confianza[, 1],
  IC_Superior_95 = intervalos_confianza[, 2],
  p_valor = resumen[, 4]
)

# Ver la tabla en la consola de R
print(tabla_or)


modelo_glm_final <- glm(DISCAPACIDAD ~ EDAD + IMC + SEDE,
                        data = datos,
                        family = binomial)

summary(modelo_glm_final)
exp(coef(modelo_glm_final))

modelo_gam_final <- gam(DISCAPACIDAD ~ s(EDAD) + s(IMC) + SEDE,
                        data = datos,
                        family = binomial)

summary(modelo_gam)
plot(modelo_gam)

gam.check(modelo_gam)
