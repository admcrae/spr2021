function err = phaseless_err(v1, v2)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
signerr = sign(v1'*v2);
err = norm(signerr*v1 - v2);
end

