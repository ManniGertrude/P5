#define analysis_cxx
#include "analysis.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TROOT.h>
#include <TRint.h>

void analysis::Loop()
{


   TH1D* Histo = new TH1D("Treffer pro Drahtnummer", "Treffer pro Drahtnummer", 48, 0.5, 48.5);

   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      
      for(UInt_t hit=0; hit<nhits_le; hit++) {
        
        // Drahtreihenfolge korrigieren
        if(wire_le[hit] % 2 == 0){wire_le[hit]--;} else{wire_le[hit]++;}

        // Rausfiltern von TOTs unter 40 ns
         if (tot[hit] < 16) break;

         //Restliche Filterung
         //if (0.884615* time_le[hit] > tot[hit]+56*0.884615)break;

        // Umrechnung der Zeiten von Messeinheiten in ns
        Double_t time=time_le[hit]*2.5;
        Double_t tot_a=tot[hit]*2.5;
        
        Histo->Fill(wire_le[hit]);
      }

   }
   Histo->GetXaxis()->SetTitle("Drahtnummer");
   Histo->GetYaxis()->SetTitle("Trefferanzahl");
   gStyle->SetOptStat(0);
   Histo->Draw();
}

int main(int argc, char** argv) {
  TROOT root("app","app");
  Int_t dargc=1;
  char** dargv = &argv[0];
  TRint *app = new TRint("app", &dargc, dargv);
  //TRint *app = new TRint("app", 0, NULL);
  TCanvas *c1 = new TCanvas("c", "c", 800, 600);
  TFile *f=new TFile(argv[1]);
  TTree *tree=(TTree*)f->FindObjectAny("t");
  //tree->Dump();
  analysis* ana = new analysis(tree);
  ana->Loop();
 
  app->Run(kTRUE);
}
