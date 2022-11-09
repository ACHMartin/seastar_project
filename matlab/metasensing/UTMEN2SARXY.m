function [X,Y] = UTMEN2SARXY(E,N,head, offset, LOOKDIR, diag)
%rotation and offset depending on the SAR look direction
%heading
%offset
% disp(' ')
% disp('BEGIN UTMEN2SARXY.m')
% disp(' ')

tamen=size(E);
if exist('diag', 'var')

    Elen=length(E);
    Nlen=length(N);
    EN=zeros(2, Elen);

    dispstat('','init'); % One time only initialization
    dispstat(sprintf(' 1D rotating...'),'keepthis','timestamp');
    for ke = 1: Elen
        %  for ky = 1: ylen
        kn=ke;
        if strcmpi(LOOKDIR, 'right')
            %XY(:, ke, kn)  = [(E(ke)-offset(1)).*sin(head) + (N(kn)-offset(2)).*cos(head);(E(ke)-offset(1)).*cos(head) - (N(kn)-offset(2))*sin(head)];
            % EN(:, kx) = [x(kx).*sin(head) + y(ky).*cos(head);x(kx).*cos(head) - y(ky).*sin(head)] + offset';
            XY(:, ke)  = [(E(ke)-offset(1)).*sin(head) + (N(kn)-offset(2)).*cos(head);(E(ke)-offset(1)).*cos(head) - (N(kn)-offset(2))*sin(head)];
        end
        if strcmpi(LOOKDIR, 'left')
            %XY(:, ke, kn)  = [(E(ke)-offset(1)).*sin(head) + (N(kn)-offset(2)).*cos(head);-(E(ke)-offset(1)).*cos(head) + (N(kn)-offset(2))*sin(head)];
            %EN(: ,kx) = [x(kx).*sin(head) - y(ky).*cos(head);x(kx).*cos(head) + y(ky).*sin(head)];
            %EN(1,kx) = EN(1, kx) + offset(1);
            %EN(2,kx) = EN(2, kx) + offset(2);
            XY(:, ke)  = [(E(ke)-offset(1)).*sin(head) + (N(kn)-offset(2)).*cos(head);-(E(ke)-offset(1)).*cos(head) + (N(kn)-offset(2))*sin(head)];
        end
        dispstat(sprintf('Progress %f%%',single(ke)/single(Elen)*100),'timestamp');
    end
    %dispstat(sprintf('Progress %f%%',single(kx)/single(xlen)*100),'timestamp');
    %end
    dispstat('.','keepprev');
    X = squeeze(XY(1,:));
    Y = squeeze(XY(2,:));


else

    if tamen(1) == 1 | tamen(2)==1

        Elen=length(E);
        Nlen=length(N);
        XY=zeros(2, Elen, Nlen);

        dispstat('','init'); % One time only initialization
        dispstat(sprintf('rotating...'),'keepthis','timestamp');
        for ke = 1: Elen
            for kn = 1: Nlen

                if strcmpi(LOOKDIR, 'right')
                    XY(:, ke, kn)  = [(E(ke)-offset(1)).*sin(head) + (N(kn)-offset(2)).*cos(head);(E(ke)-offset(1)).*cos(head) - (N(kn)-offset(2))*sin(head)];
                end
                if strcmpi(LOOKDIR, 'left')
                    XY(:, ke, kn)  = [(E(ke)-offset(1)).*sin(head) + (N(kn)-offset(2)).*cos(head);-(E(ke)-offset(1)).*cos(head) + (N(kn)-offset(2))*sin(head)];
                end

            end
            dispstat(sprintf('Progress %f%%',single(ke)/single(Elen)*100),'timestamp');
        end
        dispstat('.','keepprev');
        X = squeeze(XY(1,:, :));
        Y = squeeze(XY(2,:, :));

    end

    if tamen(1) ~= 1 & tamen(2) ~= 1

        if strcmpi(LOOKDIR, 'right')
            X  = [(E-offset(1)).*sin(head) + (N-offset(2)).*cos(head)];
            Y  = [(E-offset(1)).*cos(head) - (N-offset(2))*sin(head)];
        end
        if strcmpi(LOOKDIR, 'left')
            X  = [(E-offset(1)).*sin(head) + (N-offset(2)).*cos(head)];
            Y  = [-(E-offset(1)).*cos(head) + (N-offset(2))*sin(head)];
        end

    end

end

% disp(' ')
% disp('END UTMEN2SARXY.m')
% disp(' ')
end