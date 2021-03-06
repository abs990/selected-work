function [Eext] = getExternalEnergy(I,Wline,Wedge,Wterm)

% Eline
E_line = I;

% Eedge
[gradx, grady] = imgradientxy(I);
E_edge = -1 * (gradx .* gradx + grady .* grady);

% Eterm
 %define the masks
Cx_mask = [0 0 0; 0 -1 1; 0 0 0];
Cy_mask = [0 0 0; 0 -1 0; 0 1 0];
Cxx_mask = [0 0 0; 1 -2 1; 0 0 0];
Cyy_mask = [0 1 0; 0 -2 0; 0 1 0];
Cxy_mask = [0 -1 1;0 1 -1; 0 0 0];
 %conv
Cx = conv2(I,Cx_mask,'same');
Cy = conv2(I,Cy_mask,'same');
Cxx = conv2(I,Cxx_mask,'same');
Cyy = conv2(I,Cyy_mask,'same');
Cxy = conv2(I,Cxy_mask,'same');
 %finally compute
E_term = (Cyy .* Cx .* Cx - 2 * Cxy .* Cx .* Cy + Cxx .* Cy .* Cy)./((1 + Cx .* Cx + Cy .* Cy) .^ 1.5);

% Eext
Eext = Wline * E_line + Wedge * E_edge + Wterm * E_term; 
end
