%% clahe_mocs input: value frame
function [bestnest, fmin] = clahe_mocs_vas(img)

%% Cuckoo Search (CS) algorithm by Xin-She Yang and Suash Deb     %
% Programmed by Xin-She Yang at Cambridge University              %
% Programming dates: Nov 2008 to June 2009                        %
% Last revised: Dec  2009   (simplified version for demo only)    %
% Multiobjective cuckoo search (MOCS) added in July 2012,         %
% Then, MOCS was updated in Sept 2015.                     Thanks %
% -----------------------------------------------------------------
%% References -- Citation Details:
%% 1) X.-S. Yang, S. Deb, Cuckoo search via Levy flights,
% in: Proc. of World Congress on Nature & Biologically Inspired
% Computing (NaBIC 2009), December 2009, India,
% IEEE Publications, USA,  pp. 210-214 (2009).
% http://arxiv.org/PS_cache/arxiv/pdf/1003/1003.1594v1.pdf 
%% 2) X.-S. Yang, S. Deb, Engineering optimization by cuckoo search,
% Int. J. Mathematical Modelling and Numerical Optimisation, 
% Vol. 1, No. 4, 330-343 (2010). 
% http://arxiv.org/PS_cache/arxiv/pdf/1005/1005.2908v2.pdf
%% 3) X.-S. Yang, S. Deb, Multi-objective cuckoo search for 
% Design optimization, Computers & Operations Research, 
% vol. 40, no. 6, 1616-1624 (2013).
% ----------------------------------------------------------------%
% This demo program only implements a standard version of         %
% Cuckoo Search (CS), as the Levy flights and generation of       %
% new solutions may use slightly different methods.               %
% The pseudo code was given sequentially (select a cuckoo etc),   %
% but the implementation here uses Matlab's vector capability,    %
% which results in neater/better codes and shorter running time.  % 
% This implementation is different and more efficient than the    %
% the demo code provided in the book by 
%    "Yang X. S., Nature-Inspired Optimization Algoirthms,        % 
%     Elsevier Press, 2014.  "                                    %
% --------------------------------------------------------------- %
% =============================================================== %
%% Notes:                                                         %
% 1) The constraint-handling is not included in this demo code.   %
% The main idea to show how the essential steps of cuckoo search  %
% and multi-objective cuckoo search (MOCS) can be done.           %
% 2) Different implementations may lead to slightly different     %
% behavour and/or results, but there is nothing wrong with it,    %
% as it is the nature of random walks and all metaheuristics.     %
% --------------------------------------------------------------- %
% =============================================================== %
% This algorithm has been modified to especifically enhance images of the
% ocular conjunctiva vasculature. It uses CLAHE with Rayleigh PDF and it also 
% optimizes the alpha parameter of Rayleigh PDF.  
%
% The input image is directly the third component of the tranformed RGB to 
% HSV image.
% 
% The lower and upper bounds for MOCS have
% been selected empirically for the given images which were taken using a
% slit-lamp biomicroscope. Population size and number of iterations 
% have been adapted to the specific problem of conjunctival 
% vasculature images.
%
% The objective functions are the image conrast from the GLCM
% and the noise calculated as Fast Noise Variance Estimation (FNVE), as in
% Immerkaer J. Fast Noise Variance Estimation. 
% Comput Vis Image Underst. 1996;64(2):300-302.
% 
% 
% The Pa has been modified so it is calculated adaptive for each iteration
% according to the method described by Reda et al. (2021) in 
% A novel cuckoo search algorithm with adaptive discovery probability based 
% on double Mersenne numbers. Neural Comput & Applic 33, 16377–16402.
%
% Modified by Clara Llorens-Quintana, 2024

tic
% Population size
n = 50;
% Number of iterations
N_IterTotal = 20;
% Discovery rate of alien eggs/solutions
pa = 0.25;
% Dimensionality of the problem: number of variables
d = 4;   
% Simple lower bounds of each variable (H ntiles, Vntile, CL and alpha R)
Lb = [4 4 0.001 0.5]; 
% Simple upper bounds of each variable
Ub = [12 12 0.01 0.9];
% Number of objectives: number of functions to evaluate the search (Entropy and FNVE)
m = 2;

%% Initialize the population
Sol = zeros(n, d);
f = zeros(n, m);
for ii = 1:n
   Sol(ii,:) = Lb+(Ub-Lb).*rand(1,d); 
   f(ii,1:m) = fitness_funct_mocs(img, Sol(ii,:)); %fitness value calculated as entropy and AMBE
end
% Store the fitness or objective values
f_new = f;

%% Sort the initialized population
x = [Sol f];  % combined into a single input
% Non-dominated sorting for the initila population
Sorted = solutions_sorting_mocs(x, m, d); 
% Decompose into solutions, fitness, rank and distances
nest = Sorted(:,1:d);
f = Sorted(:,(d+1):(d+m));
RnD = Sorted(:,(d+m+1):end);
counter = 0;
%% Starting iterations
for t = 1:N_IterTotal
    % Generate new solutions (but keep the current best) - broad
     new_nest = get_cuckoos(nest,nest(1,:), Lb,Ub);

     % Discovery and randomization - local
      new_nest = empty_nests(new_nest,Lb,Ub,pa);
     
    % Evaluate this set of solutions
      for ii = 1:n
      %% Evalute the fitness/function values of the new population
        f_new(ii,1:m) = fitness_funct_mocs(img, new_nest(ii,1:d),m);
        
        if (f_new(ii,1:m) <= f(ii,1:m))  
            f(ii,1:m) = f_new(ii,1:m);
            nest(ii,:) = new_nest(ii,:);
            counter = counter + 1;
        end
        % Update the current best (stored in the first row)
        if (f_new(ii,1:m) <= f(1,1:m))
            nest(1,1:d) = new_nest(ii,1:d); 
            f(1,:) = f_new(ii,:);
        end         
      end 
      
%% Combined population consits of both the old and new solutions
%% So the total number of solutions for sorting is 2*n
%% ! It's very important to combine both populations, otherwise,
%% the results may look odd and will be very inefficient. !
       X(1:n,:) = [new_nest f_new];      % Combine new solutions
       X((n+1):(2*n),:) = [nest f];      % Combine old solutions
       Sorted = solutions_sorting_mocs(X, m, d); 
       %% Select n solutions from a combined population of 2*n solutions
       new_Sol = select_pop_mocs(Sorted, m, d, n);
       % Decompose the sorted solutions into solutions, fitness & ranking %
       nest = new_Sol(:,1:d);           % Sorted solutions/variables
       f = new_Sol(:,(d+1):(d+m));      % Sorted objective values
       RnD = new_Sol(:,(d+m+1):end);    % Sorted ranks and distances   
       bestnest = [nest(n/2, 1) nest(n/2, 2) nest(n/2, 3) nest(n/2, 4)];
        fmin = [f(n/2,1) f(n/2,2)];

end
toc
%% --------------- All subfunctions are list below ------------------     %

%% Get cuckoos by ramdom walk
function nest = get_cuckoos(nest,best,Lb,Ub)
n = size(nest,1);
% For details, please see the chapters of the following Elsevier book:  
% X. S. Yang, Nature-Inspired Optimization Algorithms, Elsevier, (2014).
beta = 3/2;  % Levy exponent in Levy flights
sigma = (gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
for jj = 1:n
    s = nest(jj,:);
    %% Levy flights by Mantegna's algorithm
    u = randn(size(s))*sigma;
    v = randn(size(s));
    step = u./abs(v).^(1/beta);
    stepsize = 0.1*step.*(s-best);
    % Now the actual random walks or flights
    s = s+stepsize.*randn(size(s));
   % Apply simple bounds/limits
   nest(jj,:) = simplebounds(s,Lb,Ub);
end

%% Replace some nests by constructing new solutions/nests
function new_nest = empty_nests(nest,Lb,Ub,pa)
% A fraction of worse nests are discovered with a probability pa
[n,d] = size(nest);
% The solutions represented by cuckoos to be discovered or not 
% with a probability pa. This action is implemented as a status vector
K = rand(size(nest))>pa; 

% New solution by biased/selective random walks
stepsize = rand*(nest(randperm(n),:)-nest(randperm(n),:));
new_nest = nest+stepsize.*K;
for jj = 1:size(new_nest,1)
    s = new_nest(jj,:);
    new_nest(jj,:) = simplebounds(s,Lb,Ub);  
end


%% Application of simple bounds
function s = simplebounds(s,Lb,Ub)
  % Apply the lower bound
  ns_tmp = s;
  I=ns_tmp<Lb;
  ns_tmp(I)=Lb(I);
  
  % Apply the upper bounds 
  J=ns_tmp>Ub;
  ns_tmp(J)=Ub(J);
  % Update this new move 
  s=ns_tmp;

  %% Select best solutions
  function new_Sol = Select_pop(nest, m, ndim, npop)
% The input population to this part has twice of the needed 
% population size (npop). Thus, selection is done based on ranking and 
% crowding distances, calculated from the non-dominated sorting
% Ranking is stored in column Krank
Krank=m+ndim+1;
% Sort the population of size 2*npop according to their ranks
[~,Index] = sort(nest(:,Krank)); 
sorted_nest = nest(Index,:);
% Get the maximum rank among the population
RankMax = max(nest(:,Krank)); 
%% Main loop for selecting solutions based on ranks and crowding distances
K = 0;  % Initialization for the rank counter 
% Loop over all ranks in the population
for ii = 1:RankMax  
    % Obtain the current rank i from sorted solutions
    RankSol = max(find(sorted_nest(:, Krank) == ii));
    % In the new cuckoos/solutions, there can be npop solutions to fill
    if RankSol < npop
       new_Sol(K+1 : RankSol, :) = sorted_nest(K+1 : RankSol, :);
    end 
    % If the population after addition is large than npop, re-arrangement
    % or selection is carried out
    if RankSol>=npop
        % Sort/Select the solutions with the current rank 
        candidate_nest = sorted_nest(K+1 : RankSol, :);
        [~,tmp_Rank] = sort(candidate_nest(:, Krank+1),'descend');
        % Fill the rest (npop-K) cuckoo/solutions up to npop solutions 
        for jj = 1:(npop-K) 
            new_Sol(K+jj, :) = candidate_nest(tmp_Rank(jj), :);
        end
    end
    % Record and update the current rank after adding new cuckoo solutions
    K = RankSol;
end
