%% Cell-wide settings

% Number of subframes. The duration of each subframe is 1ms. Each subframe 
% is also an OFDM symbol. In each subframe there are 14 time slots.

enb = lteRMCDL('R.12','FDD',1);
enb.TotSubframes = 10;

%% Propagation channel configuration

chan.NRxAnts            = 1;
chan.InitPhase          = 'Random';
chan.InitTime           = 0.0;
chan.ModelType          = 'GMEDS';
chan.NTerms             = 16;
chan.NormalizeTxAnts    = 'On';
chan.NormalizePathGains = 'On';

%% Simulation

% Number of subframes to simulate 
DurationInSubframes = 100000;
NumberUsers = 4;

for profile = {'EPA 5', 'EVA 30', 'EVA 70'}
    str = split(profile);
    chan.DelayProfile = str{1};
    chan.DopplerFreq = str2double(str{2});
    
    for MIMOcorr = {'Low', 'Medium', 'High'}
        chan.MIMOCorrelation = MIMOcorr{1};
        
        for n = 1:NumberUsers
            tic
            chan.Seed = n;
            filename = sprintf('%s%d_MIMOcorr%s_user%d', chan.DelayProfile, chan.DopplerFreq, chan.MIMOCorrelation, n);
            H = [];
            for k = 1:DurationInSubframes/enb.TotSubframes
                chan.InitTime = (k-1)*enb.TotSubframes/1000;
                toffset = 7;
                hest1 = lteDLPerfectChannelEstimate(enb,chan,[toffset,0]);
                % Assuming one pilot every 14 time slots, i.e., one pilot per subframe
                hest1 = squeeze(hest1(36,[7:14:enb.TotSubframes*14],1,[1:4]));
                H = [H; hest1];
            end
            save(filename, 'H');
            toc
        end
    end
end
        
