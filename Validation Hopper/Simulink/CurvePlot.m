% function [muscle_force, n_force_velocity, n_force_length]  =...
%     muscle_dynamics(activation,L0m,Lm,Vmax,Vmus,Fmax_mus)
%#codegen

    %Hill Type Muscle F-V relationship.  Returns a normalized force-velocity value
    
    %% FV curve
    
    Vmax=-1;
    V=-1:0.05:1.5;
    FV=zeros(size(V));
    for i=1:length(V)
        Vmus=V(i);
        if (Vmus<Vmax)
            n_force_velocity=0;
        else
            if (Vmus >= Vmax && Vmus < 0)
                n_force_velocity = ((1-(Vmus/Vmax))/(1+(Vmus/0.17/Vmax)));
            else
                n_force_velocity =(1.8-0.8*((1+(Vmus/Vmax))/(1-7.56*Vmus/0.17/Vmax)));
            end
        end
        FV(i)=n_force_velocity;
    end
    
    plot(V,FV)
    
    %Active muscle F-L relationships.  Returns a normalized F-L value

    a = 3.1108;
    b = .8698;
    s = .3914;

    L=-1:0.05:3;
    L0m=1;
    FL=zeros(size(L));
    for i=1:length(L)
        Lm=L(i);
        if Lm < 0
            Lm = 0;
        end
        n_force_length = exp(-(abs(((Lm/L0m)^b-1)/s))^a);
        FL(i)=n_force_length;
    end
    figure()
    plot(L,FL)
    
    %Total active force of muscle based on normalized F-V and F-L relationships
%     muscle_force = Fmax_mus* activation * n_force_velocity * n_force_length;
