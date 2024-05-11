
{
	UInt_t messungen = 3;
	TString histogrammName = "Driftzeiten";
	TString dateien[3] = {"2900V48THR.root", "2900V52THR.root",  "2900V56THR.root"};
	TString titel[3] = {"2900 V mit 48 Schwelle", "2900 V mit 52 Schwelle", "2900 V mit 56 Schwelle"};

	TLegend* leg = new TLegend(0.6, 0.5, 0.9, 0.7);
	leg->SetHeader("Spannungen");

	Bool_t first=true;
	UInt_t num = messungen;
	do {
		--num;
		TFile::Open(dateien[num]);
		TH1* plot = static_cast<TH1*>(gDirectory->FindObjectAny(histogrammName));
		plot->SetLineColor(num+1);
		if (first) {
			plot->Draw();
			first=false;
		} else {
			plot->Draw("same");
		}
		leg->AddEntry(plot, titel[num], "lep");
	} while (num != 0);

	leg->Draw("SAME");
}
