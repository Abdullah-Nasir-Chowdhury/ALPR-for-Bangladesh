function ThesisCode()
clear
close all
clc

deletedirforSelectedImage()
disp('function1 running:')
result = processcontrast();

if result == 0
    NET.addAssembly('System.Speech');
    mySpeaker = System.Speech.Synthesis.SpeechSynthesizer;
    mySpeaker.Rate = 1;
    Speak(mySpeaker, 'Error in Plate Detection');
    pause(1);
    Speak(mySpeaker, 'Re-evaluating');
    pause(1);
    Speak(mySpeaker, 'Changing Edge Detection Method to Grayscale, for Canny Operator');
    pause(1);    

    disp('function1.1 running:')
    result = processcontrast2();
    
    if result == 0
    NET.addAssembly('System.Speech');
    mySpeaker = System.Speech.Synthesis.SpeechSynthesizer;
    mySpeaker.Rate = 1;
    Speak(mySpeaker, 'Error in Plate Detection');
    pause(1);
    Speak(mySpeaker, 'Re-evaluating');
    pause(1);
    Speak(mySpeaker, 'Changing Edge Detection Method to Grayscale, for Canny Operator');
    pause(1);    

       disp('function1.2 running:')
        result = processcontrast3();
        
    if result == 0
    NET.addAssembly('System.Speech');
    mySpeaker = System.Speech.Synthesis.SpeechSynthesizer;
    mySpeaker.Rate = 1;
    Speak(mySpeaker, 'Error in Plate Detection');
    pause(1);
    Speak(mySpeaker, 'Re-evaluating');
    pause(1);
    Speak(mySpeaker, 'Changing Edge Detection Method to Grayscale, for Canny Operator');
    pause(1);   
        
    
disp('function2 running: ')
result = processbincanny();

        if result == 0
            NET.addAssembly('System.Speech');
            mySpeaker = System.Speech.Synthesis.SpeechSynthesizer;
            mySpeaker.Rate = 1;
            Speak(mySpeaker, 'Error in Plate Detection');
            pause(1);
            Speak(mySpeaker, 'Re-evaluating');
            pause(1);
            Speak(mySpeaker, 'Changing Edge Detection Method to Grayscale, for Canny Operator');
            pause(1);   
disp('function3 running:')
result = processgraycanny();
              
    
    clear 
    close all
    clc
    NET.addAssembly('System.Speech');
    mySpeaker = System.Speech.Synthesis.SpeechSynthesizer;
    mySpeaker.Rate = 1;
    Speak(mySpeaker, 'Error in Plate Detection');
    pause(1);
    Speak(mySpeaker, 'Re-evaluating');
    pause(1);
    Speak(mySpeaker, 'Changing Edge Detection Method to Binary, for Sobel Operator');
    pause(1);
    Speak(mySpeaker, 'Commencing Plate Detection');
    
disp('function4 running:')
reult = processbinsobel();

                    if result == 0
                      
                        NET.addAssembly('System.Speech');
                        mySpeaker = System.Speech.Synthesis.SpeechSynthesizer;
                        mySpeaker.Rate = 1;
                        Speak(mySpeaker, 'Edge Analysis Failed.')
                        pause(1);
                        Speak(mySpeaker, 'Switching Processing Techniques')
                        pause(1);
                        clear 
                        close all
                        clc

                        
disp('function5 running:')
result = processbysegmentation();

                 if result == 0
    clear 
    close all
    clc
    NET.addAssembly('System.Speech');
    mySpeaker = System.Speech.Synthesis.SpeechSynthesizer;
    mySpeaker.Rate = 1;
    Speak(mySpeaker, 'Error in Plate Detection');
    pause(1);
    Speak(mySpeaker, 'Re-evaluating');
    pause(1);
    Speak(mySpeaker, 'Changing Protocol');
    pause(1);
    Speak(mySpeaker, 'Commencing Plate Detection');

    
disp('function6 running:')
result = processmorph();

if result == 0
    
    clear 
    close all
    clc
    NET.addAssembly('System.Speech');
    mySpeaker = System.Speech.Synthesis.SpeechSynthesizer;
    mySpeaker.Rate = 1;
    Speak(mySpeaker, 'Error in Plate Detection');
    pause(1);
    Speak(mySpeaker, 'Re-evaluating');
    pause(1);
    Speak(mySpeaker, 'Changing Protocol');
    pause(1);
    Speak(mySpeaker, 'Commencing Plate Detection');
    
disp('function7 running:')
result = processerosiondilation();
        
    

                
                else
                    
                    NET.addAssembly('System.Speech');    
                    mySpeaker = System.Speech.Synthesis.SpeechSynthesizer;
                    mySpeaker.Rate = 1;
                    Speak(mySpeaker, 'Sending Plate Data');
%                     call python
               
end
                 end
                 
                    end
                     
                end
                
        end

end
end
end