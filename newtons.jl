#| echo: false
#| output: falseusing Pkg
using Pkg
Pkg.add("LDLFactorizations")
Pkg.add("Krylov")
Pkg.add("NLPModels")
Pkg.add("ADNLPModels")

using LinearAlgebra
using LDLFactorizations, Printf,Krylov 
using ADNLPModels
using NLPModels
"""
    newton_modifiee(model, eps_a=1.0e-5, eps_r=1.0e-5)

Méthode de Newton modifiée pour la minimisation d'une fonction sans contrainte.

Cette méthode cherche à minimiser une fonction objectif en utilisant les informations 
de gradient et de hessienne et la recherche linéaire d'armijo. 
La direction de descente est obtenue par une factorisation LDL modifiée de la hessienne, 
afin de garantir que la direction résultante est une direction de descente.

# Arguments
- `model` : un objet modélisant le problème.
- `eps_a` : tolérance absolue sur la norme du gradient (défaut : `1e-5`).
- `eps_r` : tolérance relative sur la norme du gradient (défaut : `1e-5`).

# Retour
- `xk` : vecteur approchant un point stationnaire de la fonction objectif.
"""
function newton_modifiee(model, x0, print_level = 0,eps_a=1.0e-6, eps_r=1.0e-6)
    xk = copy(x0)
    n = length(xk)
    fk = obj(model, xk)
    gk = grad(model, xk)
    gnorm = gnorm0 = norm(gk)
    k = 0
    if print_level ==1
    @printf "%2s  %9s  %7s  %7s %7s\n" "k" "fk" "‖grad‖" "t" "slope"
    @printf "%2d  %9.2e  %7.1e\n" k fk gnorm
    end 
    while gnorm > 1.0e-6 + 1.0e-6 * gnorm0 && k < 10*n
        Hk = hess(model,xk)
        Hk = Symmetric(triu(Hk), :U)
        ##############################
        # Factorisation LDL modéfiée #
        ##############################
        LDL = ldl_analyze(Hk)
        LDL.tol = Inf  
        LDL.r1 = 1.0e-5
        LDL = ldl_factorize!(Hk, LDL)
        ##############################
        #    Direction de descent    #
        ##############################
        dk = -LDL \ gk 
        ##############################
        #          Armijo            #
        ##############################
        slope = dot(dk, gk)
        t = 1.0
        while obj(model, xk + t .* dk) > fk + 1.0e-4 * t * slope
          t /= 2
        end
        ##############################
        #  Mise à jour des itérées   #
        ##############################
        xk .+= t .* dk
        fk = obj(model, xk)
        gk = grad(model, xk)
        gnorm = norm(gk)
        k += 1
        if print_level ==1
        @printf "%2d  %9.2e  %7.1e  %7.1e %7.1e\n" k fk gnorm t slope
        end
    end
    return xk
end


"""
    newton_inexacte(model, eps_a=1.0e-5, eps_r=1.0e-5)

Méthode de Newton inexacte pour la minimisation d'une fonction sans contrainte.

Cette méthode cherche à minimiser une fonction objectif en utilisant les informations de gradient 
et de hessienne et la recherche linéaire d'armijo. 
La direction de descente est obtenue en tronquont la méthode de gradient conjuguée 
afin de garantir que la direction résultante est une direction de descente.

# Arguments
- `model` : un objet modélisant le problème.
- `eps_a` : tolérance absolue sur la norme du gradient (défaut : `1e-5`).
- `eps_r` : tolérance relative sur la norme du gradient (défaut : `1e-5`).

# Retour
- `xk` : vecteur approchant un point stationnaire de la fonction objectif.
"""
function newton_inexacte(model, x0, print_level =0,eps_a=1.0e-6, eps_r=1.0e-6)
    xk = copy(x0)
    n = length(xk)
    fk = obj(model, xk)
    gk = grad(model, xk)
    gnorm = gnorm0 = norm(gk)
    k = 0
    if print_level ==1
    @printf "%2s  %9s  %7s  %7s %7s\n" "k" "fk" "‖grad‖" "t" "slope"
    @printf "%2d  %9.2e  %7.1e\n" k fk gnorm
    end
    while gnorm > 1.0e-6 + 1.0e-6 * gnorm0 && k < 10*n
        Hk = hess(model,xk)
        ##############################
        #    Direction de descent    #
        ##############################
        (dk, stats) = Krylov.cg(Hk, -gk,linesearch=true)
        ##############################
        #          Armijo            #
        ##############################
        slope = dot(dk, gk)
        t = 1.0
        while obj(model, xk + t .* dk) > fk + 1.0e-4 * t * slope
          t /= 2
        end
        ##############################
        #  Mise à jour des itérées   #
        ##############################
        xk .+= t .* dk
        fk = obj(model, xk)
        gk = grad(model, xk)
        gnorm = norm(gk)
        k += 1
        if print_level ==1
        @printf "%2d  %9.2e  %7.1e  %7.1e %7.1e\n" k fk gnorm t slope
        end
    end
    return xk
end
